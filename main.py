import os, time, asyncio, logging, json
from pathlib import Path
import cv2
import numpy as np
from collections import defaultdict, deque
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from os.path import basename

from app.config import *
from app.detector import Detector
from app.preprocess import gray_world, clahe_l, apply_roi, MotionGate
from app.tracker import build_tracker
from app.temporal import TemporalFilter
from app.draw import draw_tracks, encode_jpeg

# ML (risk)
from MachineLearning.pipeline_infer import RiskModel


# Load model ML
rmodel = RiskModel()

log = logging.getLogger("uvicorn.error")

# ML health & nơi ghi JSONL
APP_DIR = Path(__file__).resolve().parent
ART_DIR = Path(os.getenv("ML_ART_DIR", APP_DIR.parent / "MachineLearning" / "ml_artifacts"))
LOADED = sorted([p.name for p in ART_DIR.glob("*.pkl")]) if ART_DIR.exists() else []
EVENTS_LOG = Path(BASE_DIR) / "cv_events.jsonl"  # BASE_DIR từ config.py
log.info({"ml_artifacts_loaded": LOADED, "art_dir": str(ART_DIR)})

#Snapshot config
SNAPSHOT_DIR = Path(os.getenv("SNAPSHOT_DIR", APP_DIR / "snapshots"))
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOT_JPEG_Q = int(os.getenv("SNAPSHOT_JPEG_Q", 85))

CROPS_DIR = SNAPSHOT_DIR / "crops"
CROPS_DIR.mkdir(parents=True, exist_ok=True)

# ML RISK THRESHOLD
ML_RISK_THRESHOLD = float(os.getenv("ML_RISK_THRESHOLD", "0.85"))

#CAMERA OPEN
def open_camera(index=0):
    for backend in [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            log.info(f"Opened camera index={index} backend={backend}")
            return cap
        else:
            log.info(f"Failed open index={index} backend={backend}")
    return None


cap = open_camera(CAM_INDEX)
if cap is None:
    raise RuntimeError("Không thể mở camera.")

# ROI
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720

top = TOP_MARGIN_PX
side = SIDE_MARGIN_PX

ROI_POLY = np.array([
    [side, top],
    [W - side, top],
    [W - side, H - 1],
    [side, H - 1],
], dtype=np.int32)

# LOAD MODEL & MODULES
detector = Detector(MODEL_PATH, conf=CONF, imgsz=IMG_SIZE, iou=IOU)

# ByteTrack
tracker = build_tracker(
    TRACKER_TYPE,
    track_thresh=TRACK_THRESH,
    match_thresh=MATCH_THRESH,
    track_buffer=TRACK_BUFFER,
    frame_rate=FRAME_RATE,
    track_high_thresh=TRACK_HIGH_THRESH,
    track_low_thresh=TRACK_LOW_THRESH,
    new_track_thresh=NEW_TRACK_THRESH,
    fuse_score=FUSE_SCORE,
    proximity_thresh=PROXIMITY_THRESH,
    init_track_thresh=INIT_TRACK_THRESH,
    smooth_alpha=SMOOTH_ALPHA,
)

tfilter = TemporalFilter(
    on=HYST_ON,
    off=HYST_OFF,
    dwell=DWELL_SEC,
    max_seg_sec=10.0,
    inactivity_sec=0.8
)

mgate = MotionGate(alpha=MOTION_ALPHA, th=MOTION_THRESH, min_area=MOTION_MIN_AREA)


# trackHistory
def _iou(a, b):
    xA = max(a[0], b[0]); yA = max(a[1], b[1])
    xB = min(a[2], b[2]); yB = min(a[3], b[3])
    inter = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    areaA = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
    areaB = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
    denom = areaA + areaB - inter + 1e-6
    return inter / denom


class TrackHistory:
    def __init__(self, maxlen=300):
        self.hist = defaultdict(lambda: {
            "t": deque(maxlen=maxlen),
            "box": deque(maxlen=maxlen),  # [x1,y1,x2,y2]
            "score": deque(maxlen=maxlen),
            "cls": deque(maxlen=maxlen),
        })
        self.active_start = {}  # tid -> {"path": str, "saved": bool}

    def update(self, tracks, now, frame_for_snap=None):
        if tracks is None or len(tracks) == 0:
            return
        for x1, y1, x2, y2, tid, cls, sc in tracks:
            tid = int(tid); cls = int(cls); sc = float(sc)
            self.hist[tid]["t"].append(float(now))
            self.hist[tid]["box"].append([float(x1), float(y1), float(x2), float(y2)])
            self.hist[tid]["score"].append(sc)
            self.hist[tid]["cls"].append(cls)
            if frame_for_snap is not None and tid not in self.active_start:
                ts = int(now * 1000)
                start_path = SNAPSHOT_DIR / f"EVT-{ts}-{tid}-start.jpg"
                try:
                    cv2.imwrite(str(start_path), frame_for_snap, [int(cv2.IMWRITE_JPEG_QUALITY), SNAPSHOT_JPEG_Q])
                except Exception as e:
                    log.exception(f"Save start snapshot failed: {e}")
                self.active_start[tid] = {"path": str(start_path), "saved": True}

    def summarize(self, tid, t_enter, t_leave, frame_size=None):
        if tid not in self.hist: return None
        rec = self.hist[tid]
        T = np.array(rec["t"], dtype=float)
        if T.size == 0: return None
        sel = (T >= float(t_enter) - 1e-6) & (T <= float(t_leave) + 1e-6)
        if not np.any(sel): sel = slice(None)
        boxes = np.array(rec["box"], dtype=float)[sel]
        scores = np.array(rec["score"], dtype=float)[sel]
        if boxes.size == 0: return None

        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        area = np.clip(w, 0, None) * np.clip(h, 0, None)
        aspect = np.divide(np.clip(w, 1e-6, None), np.clip(h, 1e-6, None))
        cx = boxes[:, 0] + w / 2.0
        cy = boxes[:, 1] + h / 2.0

        if frame_size is not None:
            Hh = float(frame_size[0]); Ww = float(frame_size[1])
            cxn = np.clip(cx / max(Ww, 1.0), 0.0, 1.0)
            cyn = np.clip(cy / max(Hh, 1.0), 0.0, 1.0)
        else:
            cxn, cyn = cx, cy

        ious = []
        for i in range(1, boxes.shape[0]):
            ious.append(_iou(boxes[i - 1], boxes[i]))
        iou_stab = float(np.mean(ious)) if ious else 0.0

        speeds = []
        for i in range(1, boxes.shape[0]):
            dx = cx[i] - cx[i - 1]; dy = cy[i] - cy[i - 1]
            speeds.append(float(np.hypot(dx, dy)))
        mean_speed = float(np.mean(speeds)) if speeds else 0.0

        return {
            "score_mean": float(np.mean(scores)),
            "score_std": float(np.std(scores)),
            "score_max": float(np.max(scores)),
            "area_mean": float(np.mean(area)),
            "area_std": float(np.std(area)),
            "aspect_mean": float(np.mean(aspect)),
            "aspect_std": float(np.std(aspect)),
            "cx_mean": float(np.mean(cxn)),
            "cy_mean": float(np.mean(cyn)),
            "iou_stability": iou_stab,
            "mean_speed": mean_speed,
        }

    def pop_start_snapshot(self, tid):
        meta = self.active_start.get(int(tid))
        return meta["path"] if meta and meta.get("saved") else None

    def clear_track(self, tid):
        tid = int(tid)
        if tid in self.hist: del self.hist[tid]
        if tid in self.active_start: del self.active_start[tid]


thist = TrackHistory(maxlen=300)

app = FastAPI()

try:
    app.mount("/snapshots", StaticFiles(directory=str(SNAPSHOT_DIR)), name="snapshots")
    log.info(f"Mounted snapshots dir at /snapshots -> {SNAPSHOT_DIR}")
except Exception as e:
    log.warning(f"Mount snapshots failed: {e}")

HTML = """
<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8"/>
<title>PHÁT HIỆN GIAN LẬN</title>
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans:wght@300;400;700;800&display=swap" rel="stylesheet">
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<style>
html,body{height:100%;margin:0;overflow:hidden;}
 body{
    font-family: "Noto Sans", "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    -webkit-font-smoothing:antialiased;
    -moz-osx-font-smoothing:grayscale;
    background:#000;
    color:#eee;
  }
#bar{padding:8px 12px;background:#111;display:flex;align-items:center}
#fps{margin-left:auto;color:#0f0}
#btnToggle{
    margin-left:12px;
    padding:6px 12px;
    border-radius:8px;
    border:0;
    background:#1e90ff;
    color:white;
    font-weight:600;
    cursor:pointer;
    font-size:14px;
}
img{width:100%;height:calc(100vh - 40px);object-fit:contain;display:block}

:root{ --bg-page:#ffffff; --text-dark:#0f172a; --muted:#6b7280; --accent-blue:#0b84ff; --accent-green:#0dd6a2; --container-max-width:1600px; --tbl-font:15px; --th-font:16px; --thumb-w:160px; --thumb-h:96px; --modal-max-w:1000px; --modal-max-h:86vh; }

#overlay{ position:fixed; left:0; width:100%; background:var(--bg-page); z-index:900; display:none; overflow:auto; -webkit-overflow-scrolling:touch; }

#overlay,
#overlay .panel,
#overlay .wrap,
#overlay .tableWrap,
#overlay table,
#overlay thead th,
#overlay tbody td,
#overlay .panel-header h2,
#overlay .panel-sub,
#overlay .pill,
#overlay .empty,
#jsonModal pre,
#modal { 
  color: #000 !important;
}

#overlay thead th {
  color: #000 !important;
}

#overlay .pill {
  color: #000 !important;
  background: #f3f4f6;
}

.wrap{max-width:var(--container-max-width);margin:18px auto;padding:0 20px}
.panel{padding:18px}
.panel-header{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.panel-header h2{margin:0;font-size:20px;font-weight:800;color:var(--text-dark)}
.panel-sub{font-size:13px;color:var(--muted)}
.controls{margin-left:auto;display:flex;gap:10px;align-items:center}
.controls select,.controls input{padding:8px 10px;border:1px solid #e6e9ee;border-radius:10px;font-size:14px;min-width:200px}
.tableWrap{margin-top:14px;border:1px solid #eef2f6;border-radius:10px;overflow:hidden;background:white}
table{width:100%;border-collapse:collapse;font-size:var(--tbl-font);table-layout:fixed}
thead th{position:sticky;top:0;background:#fbfdff;padding:12px 12px;border-bottom:1px solid #eef2f6;font-weight:800;text-align:left;cursor:pointer;font-size:var(--th-font)}
.sort-ind{display:inline-block;margin-left:8px;color:var(--muted);font-weight:700;font-size:14px;width:18px;text-align:center}

thead th:nth-child(1), tbody td:nth-child(1){width:14%}
thead th:nth-child(2), tbody td:nth-child(2){width:12%}
thead th:nth-child(3), tbody td:nth-child(3){width:12%}
thead th:nth-child(4), tbody td:nth-child(4){width:12%}
thead th:nth-child(5), tbody td:nth-child(5){width:12%}
thead th:nth-child(6), tbody td:nth-child(6){width:13%}
thead th:nth-child(7), tbody td:nth-child(7){width:13%}
thead th:nth-child(8), tbody td:nth-child(8){width:12%}
tbody td{padding:12px 12px;border-bottom:1px solid #fbfbff;vertical-align:middle;color:var(--text-dark)}
tbody tr:hover{background:#fbfbff}
.ts-time{font-size:16px;font-weight:700;color:var(--text-dark);line-height:1}
.ts-date{font-size:12px;color:var(--muted);margin-top:6px;line-height:1}
.pill{padding:5px 8px;background:#f3f4f6;border-radius:999px;font-weight:700;font-size:13px;color:#374151;display:inline-block}
.thumb{ width:var(--thumb-w); max-width:100%; height:var(--thumb-h); max-height:100%; border-radius:10px; border:1px solid #eef2f6; object-fit:cover; cursor:pointer; display:block; margin:6px auto; background:#fff; }
td.actions-cell { padding: 0 !important; vertical-align: middle; }
.btn-view{ padding:8px 10px; border-radius:10px; border:0; font-size:13px; font-weight:800; cursor:pointer; background:var(--accent-green); color:#000; box-shadow:none; min-width:88px; max-width:100%; margin:0; line-height:1.05; display:inline-flex; align-items:center; justify-content:center; }
.empty{padding:18px;text-align:center;color:var(--muted);font-size:14px}
#backdrop{display:none;position:fixed;left:0;top:0;width:100%;height:100%;background:rgba(0,0,0,0.45);z-index:1000}
#modal{display:none;position:fixed;left:50%;top:50%;transform:translate(-50%,-50%) scale(.98);background:#fff;padding:14px;border-radius:12px;box-shadow:0 30px 80px rgba(0,0,0,0.28);z-index:1010;max-width:var(--modal-max-w);width:calc(min(96vw,var(--modal-max-w)));max-height:var(--modal-max-h);overflow:auto;transition:transform .18s ease, opacity .18s ease;opacity:0}
#modal.show{transform:translate(-50%,-50%) scale(1);opacity:1}
#modalImg{display:block;width:100%;height:auto;max-height:calc(var(--modal-max-h) - 24px);border-radius:10px;object-fit:contain}
#jsonBackdrop{display:none;position:fixed;left:0;top:0;width:100%;height:100%;background:rgba(0,0,0,0.45);z-index:1020}
#jsonModal{display:none;position:fixed;left:50%;top:50%;transform:translate(-50%,-50%);background:#fff;padding:18px;border-radius:10px;box-shadow:0 30px 80px rgba(0,0,0,0.28);z-index:1030;max-width:900px;width:92vw;max-height:80vh;overflow:auto}
#jsonModal pre{white-space:pre-wrap;word-break:break-word;font-size:13px;color:#0b1720}
.jsonClose{display:inline-block;margin-top:12px;padding:8px 12px;border-radius:8px;background:#e6eefc;border:0;cursor:pointer}

@media (max-width:900px){
 :root{--thumb-w:120px;--thumb-h:76px}
 thead th:nth-child(1), tbody td:nth-child(1){width:30%}
 thead th:nth-child(6), tbody td:nth-child(6), thead th:nth-child(7), tbody td:nth-child(7){width:22%}
 #btnToggle{padding:6px 10px;font-size:13px}
}
</style>
</head>
<body>

<div id="bar">
    <strong>Phát hiện vật vi phạm trong thi online</strong>
    <span id="fps"></span>
    <button id="btnToggle">Xem vi phạm</button>
</div>

<img id="img" alt="camera preview" />

<div id="overlay" aria-hidden="true">
  <div class="panel wrap">
    <div class="panel-header">
      <div>
        <h2>Bảng vi phạm</h2>
      </div>
      <div class="controls">
        <select id="clsFilter"><option value="">Tất cả</option></select>
        <input id="textFilter" placeholder="Tìm kiếm" />
      </div>
    </div>

    <div class="tableWrap" role="table" aria-label="Violation events">
        <table id="logTable">
            <thead>
                <tr>
                    <th data-key="t_enter">Thời gian <span class="sort-ind" id="si-t_enter"></span></th>
                    <th data-key="cls_name">Vật vi phạm <span class="sort-ind" id="si-cls_name"></span></th>
                    <th data-key="tid">Mã theo dõi <span class="sort-ind" id="si-tid"></span></th>
                    <th data-key="features.dwell">thời lượng(s) <span class="sort-ind" id="si-features.dwell"></span></th>
                    <th data-key="ml.risk">Mức độ rủi ro <span class="sort-ind" id="si-ml.risk"></span></th>
                    <th>Ảnh bắt đầu</th>
                    <th>Ảnh kết thúc</th>
                    <th>Chi tiết</th>
                </tr>
            </thead>
        <tbody id="logBody">
          <tr><td colspan="8" class="empty">No events yet</td></tr>
        </tbody>
      </table>
    </div>
  </div>
</div>

<div id="backdrop" aria-hidden="true"></div>
<div id="modal" role="dialog" aria-hidden="true"><img id="modalImg" alt="snapshot large"></div>

<div id="jsonBackdrop" aria-hidden="true"></div>
<div id="jsonModal" role="dialog" aria-hidden="true">
  <h3 style="margin:0 0 8px 0">Event JSON (Processed)</h3>
  <pre id="jsonContent">{}</pre>
  <div style="text-align:right"><button class="jsonClose" id="jsonClose">Close</button></div>
</div>

<script>
function fmtTsParts(ts){ try{ const d=new Date(Math.round(ts*1000)); return {time:d.toLocaleTimeString(), date:d.toLocaleDateString()}; }catch(e){ return {time:String(ts), date:""} } }

const camImg = document.getElementById("img");
const fpsEl = document.getElementById("fps");
const overlay = document.getElementById("overlay");
const btnToggle = document.getElementById("btnToggle");
const clsFilter = document.getElementById("clsFilter");
const textFilter = document.getElementById("textFilter");
const logBody = document.getElementById("logBody");
const backdrop = document.getElementById("backdrop");
const modal = document.getElementById("modal");
const modalImg = document.getElementById("modalImg");
const jsonBackdrop = document.getElementById("jsonBackdrop");
const jsonModal = document.getElementById("jsonModal");
const jsonContent = document.getElementById("jsonContent");
const jsonClose = document.getElementById("jsonClose");

let eventsCache = [], sortState = {key: "t_enter", dir: "desc"};
let ws;

const RISK_THRESHOLD = 0.85;

function resolveSnapshotUrl(url){ if(!url) return ""; if(url.startsWith("http") || url.startsWith("/")) return url; return "/snapshots/" + url; }

function connectWS(){
  try{
    ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + "/ws");
  }catch(e){
    console.warn("WS connect failed", e);
    return;
  }
  ws.onopen = () => console.log("WS connected");
  ws.onmessage = (ev) => {
    try{
      const m = JSON.parse(ev.data);
      if(m.type === "frame"){
        if(m.data) camImg.src = "data:image/jpeg;base64," + m.data;
        if(typeof m.fps !== "undefined") fpsEl.textContent = Math.round(m.fps) + " fps";
      } else if(m.type === "event"){
        // Nhận event từ backend (đã có ML data), lưu vào cache
        if(m.event){ eventsCache.unshift(m.event); if(eventsCache.length > 2000) eventsCache.length = 2000; refreshClassOptions(); renderTable(); }
      }
    }catch(e){ console.warn("WS message parse error", e); }
  };
  ws.onclose = () => setTimeout(connectWS, 1000);
  ws.onerror = (e) => console.warn("WS err", e);
}
connectWS();

function openModal(url){ if(!url) return; const r = resolveSnapshotUrl(url); if(!r) return; modalImg.src = r; backdrop.style.display = "block"; modal.style.display = "block"; setTimeout(()=> modal.classList.add("show"), 16); backdrop.setAttribute("aria-hidden", "false"); modal.setAttribute("aria-hidden", "false"); }
function closeModal(){ modal.classList.remove("show"); setTimeout(()=>{ modalImg.src = ""; backdrop.style.display = "none"; modal.style.display = "none"; backdrop.setAttribute("aria-hidden", "true"); modal.setAttribute("aria-hidden", "true"); }, 180); }
backdrop.addEventListener("click", closeModal);
modal.addEventListener("click", (e) => e.stopPropagation());

function openJsonModal(ev){ if(!ev) return; jsonContent.textContent = JSON.stringify(ev, null, 2); jsonBackdrop.style.display = "block"; jsonModal.style.display = "block"; jsonBackdrop.setAttribute("aria-hidden", "false"); jsonModal.setAttribute("aria-hidden", "false"); }
function closeJsonModal(){ jsonBackdrop.style.display = "none"; jsonModal.style.display = "none"; jsonBackdrop.setAttribute("aria-hidden", "true"); jsonModal.setAttribute("aria-hidden", "true"); jsonContent.textContent = ""; }
jsonBackdrop.addEventListener("click", closeJsonModal);
jsonClose.addEventListener("click", closeJsonModal);
document.addEventListener("keydown",(e)=>{ if(e.key === "Escape"){ if(modal.style.display === "block") closeModal(); if(jsonModal.style.display === "block") closeJsonModal(); } });

function valueByKey(ev,key){ if(!ev) return null; const parts = key.split("."); let cur = ev; for(const p of parts){ if(cur == null) return null; cur = cur[p]; } return cur; }

function renderTable(){
  let arr = eventsCache.slice();
  const q = (textFilter.value || "").toLowerCase().trim();
  if(q) arr = arr.filter(e => ((e.event_id||"") + " " + (e.tid||"") + " " + (e.cls_name||"")).toLowerCase().includes(q));
  const cf = clsFilter.value;
  if(cf) arr = arr.filter(e => String(e.cls_name || e.cls) === String(cf));

  // RISK FILTER
  arr = arr.filter(e => {
      if (!e.ml || typeof e.ml.risk === 'undefined') return true; 
      return e.ml.risk >= RISK_THRESHOLD;
  });

  arr.sort((a,b) => {
    const va = valueByKey(a, sortState.key); const vb = valueByKey(b, sortState.key);
    const na = Number(va), nb = Number(vb);
    if(!isNaN(na) && !isNaN(nb)) return sortState.dir === "asc" ? na - nb : nb - na;
    const sa = (va == null ? "" : String(va)); const sb = (vb == null ? "" : String(vb));
    return sortState.dir === "asc" ? sa.localeCompare(sb) : sb.localeCompare(sa);
  });

  logBody.innerHTML = "";
  if(!arr.length){ logBody.innerHTML = `<tr><td colspan="8" class="empty">No high-risk events yet</td></tr>`; return; }

  for(const ev of arr){
    const tr = document.createElement("tr");
    const t_ts = document.createElement("td"); const parts = fmtTsParts(ev.t_enter); t_ts.innerHTML = `<div class="ts-time">${parts.time}</div><div class="ts-date">${parts.date}</div>`; tr.appendChild(t_ts);
    const t_cls = document.createElement("td"); t_cls.innerHTML = `<span class="pill">${ev.cls_name || ev.cls || "?"}</span>`; tr.appendChild(t_cls);
    const t_tid = document.createElement("td"); t_tid.textContent = ev.tid || "-"; tr.appendChild(t_tid);
    const t_dwell = document.createElement("td"); t_dwell.textContent = ev.features && ev.features.dwell ? Number(ev.features.dwell).toFixed(2) : "-"; tr.appendChild(t_dwell);

    // Hiển thị Risk thay vì score trung bình (cũ)
    const t_risk = document.createElement("td"); 
    if (ev.ml && typeof ev.ml.risk !== 'undefined') {
        const r = Number(ev.ml.risk);
        t_risk.textContent = r.toFixed(2);
        t_risk.style.fontWeight = "bold";
        t_risk.style.color = r > 0.85 ? "red" : "orange";
    } else {
        t_risk.textContent = "-";
    }
    tr.appendChild(t_risk);

    const t_start = document.createElement("td");
    if(ev.snapshots && ev.snapshots.start){ const startUrl = resolveSnapshotUrl(ev.snapshots.start); const im = document.createElement("img"); im.className = "thumb"; im.src = startUrl; im.alt = "start"; im.addEventListener("click", (e) => { e.stopPropagation(); openModal(startUrl); }); t_start.appendChild(im); }
    else { t_start.innerHTML = "<div class='empty'>-</div>"; }
    tr.appendChild(t_start);

    const t_end = document.createElement("td");
    if(ev.snapshots && ev.snapshots.end){ const endUrl = resolveSnapshotUrl(ev.snapshots.end); const im2 = document.createElement("img"); im2.className = "thumb"; im2.src = endUrl; im2.alt = "end"; im2.addEventListener("click", (e) => { e.stopPropagation(); openModal(endUrl); }); t_end.appendChild(im2); }
    else { t_end.innerHTML = "<div class='empty'>-</div>"; }
    tr.appendChild(t_end);

    const t_act = document.createElement("td"); t_act.className = "actions actions-cell";
    const vbtn = document.createElement("button"); vbtn.className = "btn-view"; vbtn.textContent = "Xem JSON"; vbtn.addEventListener("click", ()=>{ openJsonModal(ev); });
    t_act.appendChild(vbtn); tr.appendChild(t_act);

    logBody.appendChild(tr);
  }
}

function refreshClassOptions(){ const s = new Set(eventsCache.map(e => e.cls_name || e.cls)); clsFilter.innerHTML = `<option value="">Tất cả</option>`; Array.from(s).sort().forEach(c => { const o = document.createElement("option"); o.value = c; o.textContent = c; clsFilter.appendChild(o); }); }
document.querySelectorAll("#logTable thead th[data-key]").forEach(th => { th.addEventListener("click", () => { const key = th.getAttribute("data-key"); if(sortState.key === key) sortState.dir = (sortState.dir === "asc") ? "desc" : "asc"; else { sortState.key = key; sortState.dir = "asc"; } updateSortIcons(); renderTable(); }); });
function updateSortIcons(){ document.querySelectorAll("#logTable thead th[data-key]").forEach(th => { const key = th.getAttribute("data-key"); const span = th.querySelector(".sort-ind"); if(!span) return; if(sortState.key === key){ span.textContent = (sortState.dir === "asc") ? "▲" : "▼"; span.classList.add("sort-active"); } else { span.textContent = ""; span.classList.remove("sort-active"); } }); }
clsFilter.addEventListener("change", renderTable);
textFilter.addEventListener("input", () => renderTable());

function openOverlay(){
  const bar = document.getElementById("bar");
  const h = (bar && bar.offsetHeight) ? bar.offsetHeight : 40; // fallback 40px
  overlay.style.top = h + "px";
  overlay.style.height = `calc(100vh - ${h}px)`;
  overlay.style.display = "block";
  overlay.setAttribute("aria-hidden", "false");
  refreshClassOptions(); updateSortIcons(); renderTable();

  if(!window._violationRefreshTimer){
    window._violationRefreshTimer = setInterval(()=>{
      if(overlay.style.display === "block"){
        fetch('/api/events').then(r => r.json()).then(data => {
          const existing = new Set(eventsCache.map(e => e.event_id));
          const added = [];
          data.reverse().forEach(ev => {
            if(!existing.has(ev.event_id)){ eventsCache.unshift(ev); added.push(ev.event_id); }
          });
          if(added.length){ refreshClassOptions(); renderTable(); }
        }).catch(()=>{});
      }
    }, 2000);
  }
}

function closeOverlay(){
  overlay.style.display = "none";
  overlay.setAttribute("aria-hidden", "true");
  closeModal();
  closeJsonModal();
}

btnToggle.addEventListener("click", () => {
  if(overlay.style.display === "block") {
    closeOverlay();
    btnToggle.textContent = "Xem vi phạm";
  } else {
    openOverlay();
    btnToggle.textContent = "Ẩn Vi phạm";
  }
});

refreshClassOptions(); updateSortIcons(); renderTable();

</script>
</body>
</html>
"""


@app.get("/")
def index():
    return HTMLResponse(HTML)

@app.get("/json-flow")
def json_flow_page():
    page_path = os.path.join(os.path.dirname(__file__), "json_flow.html")
    if os.path.exists(page_path):
        with open(page_path, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>Missing json_flow.html</h1>", status_code=404)

@app.get("/api/events")
def api_events(limit: int = 50):
    out = []
    try:
        if EVENTS_LOG.exists():
            with open(EVENTS_LOG, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
            sel = lines[-limit:]
            for l in sel:
                try:
                    ev = json.loads(l)
                    # normalize snapshot paths to basename so client can load /snapshots/<basename>
                    if "snapshots" in ev:
                        s = ev.get("snapshots", {})
                        if s.get("start"):
                            ev["snapshots"]["start"] = basename(str(s.get("start")))
                        if s.get("end"):
                            ev["snapshots"]["end"] = basename(str(s.get("end")))
                    out.append(ev)
                except Exception:
                    continue
    except Exception as e:
        log.exception(f"api_events error: {e}")
    return JSONResponse(out)


@app.websocket("/ws")
async def ws_stream(ws: WebSocket):
    await ws.accept()
    log.info("WebSocket connected")

    last_detect_t = 0.0
    last_probe_t = 0.0
    t_prev_fps = time.time()
    sent_frames = 0
    last_risk_value = None  # overlay HUD

    # giữ tên lớp của lần detect gần nhất (để ghi cls_name)
    last_names = {}

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                await asyncio.sleep(0.01)
                continue

            # === Preprocess: WB + CLAHE ===
            if USE_GRAYWORLD:
                frame = gray_world(frame)
            if USE_CLAHE:
                frame = clahe_l(frame)

            # === ROI mask + Motion gate ===
            frame_roi, _ = apply_roi(frame, ROI_POLY)
            is_motion = mgate.check(frame_roi)

            # --- Blur proxy (độ nét) trên ROI
            roi_gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
            roi_blur_var = float(cv2.Laplacian(roi_gray, cv2.CV_64F).var())

            now = time.time()

            # --- detect thường xuyên khi đang có track sống ---
            has_live_tracks = bool(getattr(tracker, "_id_cls", {}))

            need_detect = False
            if has_live_tracks:
                need_detect = True
            else:
                if is_motion and (now - last_detect_t) >= (1.0 / DETECT_FPS):
                    need_detect = True
                elif (now - last_probe_t) >= (1.0 / PROBE_FPS):
                    need_detect = True
                    last_probe_t = now

            dets = None; names = {}
            if need_detect:
                dets, names = detector.infer(frame_roi)
                if isinstance(names, dict):
                    last_names = names
                last_detect_t = now

            # === Track every frame ===
            tracks = tracker.update(
                np.array(dets, dtype=float) if dets else None,
                frame.shape
            )

            # Lọc box invalid
            if len(tracks):
                tracks = np.array([t for t in tracks if t[0] < t[2] and t[1] < t[3]], dtype=object)

            # Cập nhật lịch sử
            thist.update(tracks, now, frame_for_snap=frame)

            # === Temporal filter & events ===
            events = tfilter.update(tracks)

            # === Khi event kết thúc: ghi JSONL + snapshot end + gọi ML ===
            if events:
                for ev in events:
                    dwell_sec = float(ev["t_leave"] - ev["t_enter"])
                    score_avg = float(ev.get("score_avg", 0.0))
                    motion_px = int(getattr(mgate, "last_motion_px", 0))
                    blur_mean = float(roi_blur_var)

                    tid = int(ev["id"])
                    cls_id = int(ev.get("cls", -1))
                    cls_name = last_names.get(cls_id, str(cls_id)) if isinstance(last_names, dict) else str(cls_id)

                    # snapshot
                    start_path = thist.pop_start_snapshot(tid)
                    eid = f"EVT-{int(ev['t_enter'] * 1000)}-{tid}-cls{cls_id}"
                    end_path = SNAPSHOT_DIR / f"{eid}-end.jpg"
                    try:
                        cv2.imwrite(str(end_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), SNAPSHOT_JPEG_Q])
                    except Exception as e:
                        log.exception(f"Save end snapshot failed: {e}")

                    # thống kê khoảng [t_enter, t_leave]
                    stats = thist.summarize(tid, ev["t_enter"], ev["t_leave"], frame_size=frame.shape[:2]) or {}

                    feats_event = {
                        "dwell": dwell_sec,
                        "score_mean": float(stats.get("score_mean", score_avg)),
                        "score_std": float(stats.get("score_std", 0.0)),
                        "score_max": float(stats.get("score_max", score_avg)),
                        "area_mean": float(stats.get("area_mean", 0.0)),
                        "area_std": float(stats.get("area_std", 0.0)),
                        "aspect_mean": float(stats.get("aspect_mean", 0.0)),
                        "aspect_std": float(stats.get("aspect_std", 0.0)),
                        "cx_mean": float(stats.get("cx_mean", 0.0)),
                        "cy_mean": float(stats.get("cy_mean", 0.0)),
                        "iou_stability": float(stats.get("iou_stability", 0.0)),
                        "mean_speed": float(stats.get("mean_speed", 0.0)),
                        "motion_mean": float(motion_px),
                        "motion_peak": float(motion_px),
                        "blur_mean": blur_mean,
                    }

                    # crop object from start snapshot
                    crop_rel_path = None
                    try:
                        # choose source image for crop: prefer start snapshot, else end snapshot
                        src_img_path = None
                        if start_path:
                            src_img_path = Path(start_path)
                        else:
                            src_img_path = end_path

                        img = None
                        if src_img_path and src_img_path.exists():
                            img = cv2.imread(str(src_img_path))

                        # compute bbox: prefer representative bbox from stats (repr_box), else derive from cx/area/aspect
                        bbox = None
                        if stats.get("repr_box"):
                            rb = stats.get("repr_box")
                            bbox = [int(rb[0]), int(rb[1]), int(rb[2]), int(rb[3])]
                        else:
                            try:
                                rec = thist.hist.get(int(tid), None)
                                if rec is not None:
                                    T = np.array(rec["t"], dtype=float)
                                    sel = (T >= float(ev["t_enter"]) - 1e-6) & (T <= float(ev["t_leave"]) + 1e-6)
                                    boxes_arr = np.array(rec["box"], dtype=float)[sel]
                                    scores_arr = np.array(rec["score"], dtype=float)[sel]
                                    if boxes_arr.size:
                                        idx_best = int(np.argmax(scores_arr)) if scores_arr.size > 0 else 0
                                        b = boxes_arr[idx_best]
                                        bbox = [int(b[0]), int(b[1]), int(b[2]), int(b[3])]
                            except Exception:
                                bbox = None

                        # if still no bbox, compute from cx_mean/cy_mean/area_mean/aspect_mean
                        if bbox is None or img is None:
                            try:
                                am = float(stats.get("area_mean", 0.0))
                                asp = float(stats.get("aspect_mean", 0.0))
                                cxn = float(stats.get("cx_mean", 0.5))
                                cyn = float(stats.get("cy_mean", 0.5))
                                Hh = float(frame.shape[0]); Ww = float(frame.shape[1])
                                if am > 0 and asp > 0:
                                    w = float(np.sqrt(max(am * asp, 1.0)))
                                    h = float(max(am / w, 1.0))
                                else:
                                    w, h = 96.0, 96.0
                                cx_abs = cxn * Ww
                                cy_abs = cyn * Hh
                                x1 = int(max(0, cx_abs - w / 2.0))
                                y1 = int(max(0, cy_abs - h / 2.0))
                                x2 = int(min(Ww - 1, x1 + w))
                                y2 = int(min(Hh - 1, y1 + h))
                                bbox = [x1, y1, x2, y2]
                                if img is None and src_img_path and src_img_path.exists():
                                    img = cv2.imread(str(src_img_path))
                            except Exception:
                                bbox = None

                        # perform crop if possible
                        if img is not None and bbox is not None:
                            x1, y1, x2, y2 = bbox
                            H_img, W_img = img.shape[0], img.shape[1]
                            x1 = max(0, min(int(x1), W_img - 1))
                            x2 = max(0, min(int(x2), W_img - 1))
                            y1 = max(0, min(int(y1), H_img - 1))
                            y2 = max(0, min(int(y2), H_img - 1))
                            if x2 > x1 and y2 > y1:
                                crop = img[y1:y2, x1:x2]
                            else:
                                crop = None
                        else:
                            crop = None

                        # fallback: if crop None but img available, create small centered crop
                        if crop is None and img is not None:
                            H_img, W_img = img.shape[0], img.shape[1]
                            cx_c = W_img // 2; cy_c = H_img // 2
                            w, h = 128, 128
                            x1 = max(0, cx_c - w // 2); y1 = max(0, cy_c - h // 2)
                            x2 = min(W_img - 1, x1 + w); y2 = min(H_img - 1, y1 + h)
                            crop = img[y1:y2, x1:x2]

                        # save crop if we have it
                        if crop is not None:
                            crop_fname = f"{eid}.jpg"
                            crop_path = CROPS_DIR / crop_fname
                            try:
                                cv2.imwrite(str(crop_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), SNAPSHOT_JPEG_Q])
                                crop_rel_path = f"crops/{crop_fname}"
                            except Exception as e:
                                log.exception(f"Save crop failed: {e}")

                    except Exception as e:
                        log.exception(f"Crop generation error: {e}")

                    # Tạo object event
                    event_row = {
                        "event_id": eid,
                        "tid": tid,
                        "cls": cls_id,
                        "cls_name": cls_name,
                        "t_enter": float(ev["t_enter"]),
                        "t_leave": float(ev["t_leave"]),
                        "features": feats_event,
                        "snapshots": {
                            "start": basename(str(start_path)) if start_path else None,
                            "end": basename(str(end_path)),
                            "crop": crop_rel_path
                        }
                    }

                    # Chạy ML Inference
                    ml_out = {"p_rf": None, "risk": None, "kmeans": {"cluster_id": None, "dist_min": None}}
                    try:
                        if rmodel is not None:
                            ml_out = rmodel.predict(event_row)
                        else:
                            ml_out = {
                                "p_rf": 0.0,
                                "risk": 0.0,
                                "kmeans": {
                                    "cluster_id": None,
                                    "dist_min": None
                                }
                            }
                        last_risk_value = ml_out.get("risk", None)
                    except Exception as e:
                        log.exception(f"ML inference error: {e}")

                    #Lấy risk và quyết định giữ hay bỏ event
                    risk_val = ml_out.get("risk", 0.0)
                    try:
                        risk_val = float(risk_val) if risk_val is not None else 0.0
                    except Exception:
                        risk_val = 0.0

                    event_row["ml"] = ml_out

                    # Lọc rish:
                    if risk_val >= ML_RISK_THRESHOLD:
                        try:
                            with open(EVENTS_LOG, "a", encoding="utf-8") as f:
                                f.write(json.dumps(event_row, ensure_ascii=False) + "\n")
                        except Exception as e:
                            log.exception(f"Write JSONL failed: {e}")

                        try:
                            await ws.send_json({"type": "event", "event": event_row})
                        except Exception as e:
                            log.exception("ws send event failed")

                    # xóa track history
                    thist.clear_track(tid)

            # raw & overlay risk & gửi
            vis = draw_tracks(frame.copy(), tracks, names=names, roi_poly=ROI_POLY)

            # HUD risk với màu
            if last_risk_value is not None:
                try:
                    col = (0, 255, 0) if last_risk_value < 0.4 else (0, 255, 255) if last_risk_value < 0.7 else (0, 0, 255)
                    cv2.putText(vis, f"risk={last_risk_value:.2f}", (12, 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2, cv2.LINE_AA)
                except Exception:
                    pass

            # FPS
            sent_frames += 1
            dt = now - t_prev_fps
            if dt <= 0: dt = 1e-6
            send_fps = sent_frames / dt

            b64 = encode_jpeg(vis, JPEG_Q)
            if b64:
                await ws.send_json({"type": "frame", "data": b64, "fps": send_fps})

            await asyncio.sleep(max(0.0, 1.0 / MAX_FPS))

    except WebSocketDisconnect:
        log.info("WebSocket disconnected")
    except Exception as e:
        log.exception(f"WS error: {e}")


if __name__ == "__main__":
    import webbrowser
    import threading

    log.info(f"Starting server | Model: {MODEL_PATH}")

    # Tự động mở trang web sau khi server khởi động
    threading.Timer(
        1.5,
        lambda: webbrowser.open("http://127.0.0.1:8010/")
    ).start()

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8010,
        reload=False
    )