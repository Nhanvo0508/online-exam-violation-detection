import numpy as np

try:
    from app.config import (
        TRACK_THRESH as CFG_TRACK_THRESH,
        MATCH_THRESH as CFG_MATCH_THRESH,
        TRACK_BUFFER as CFG_TRACK_BUFFER,
        FRAME_RATE   as CFG_FRAME_RATE,
    )
except Exception:
    CFG_TRACK_THRESH = 0.75
    CFG_MATCH_THRESH = 0.80
    CFG_TRACK_BUFFER = 10
    CFG_FRAME_RATE   = 30

def _get_cfg_opt(name, default=None):
    try:
        from app import config as _cfg
        return getattr(_cfg, name, default)
    except Exception:
        return default

CFG_SMOOTH_ALPHA   = _get_cfg_opt("SMOOTH_ALPHA", None)         # "", "0" -> tắt
CFG_TRACK_LOW_TH   = _get_cfg_opt("TRACK_LOW_THRESH", 0.10)     # nếu muốn cấu hình riêng
CFG_TRACK_HIGH_TH  = _get_cfg_opt("TRACK_HIGH_THRESH", None)    # None => dùng track_thresh
CFG_NEW_TRACK_TH   = _get_cfg_opt("NEW_TRACK_THRESH", None)     # None => dùng track_thresh
CFG_FUSE_SCORE     = bool(_get_cfg_opt("FUSE_SCORE", True))
CFG_PROXIMITY_TH   = float(_get_cfg_opt("PROXIMITY_THRESH", 0.5))
CFG_INIT_TRACK_TH  = _get_cfg_opt("INIT_TRACK_THRESH", None)    # None => dùng track_high_thresh

# helpers
def iou(a, b):
    xA = max(a[0], b[0]); yA = max(a[1], b[1])
    xB = min(a[2], b[2]); yB = min(a[3], b[3])
    inter = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    areaA = (a[2]-a[0]+1)*(a[3]-a[1]+1)
    areaB = (b[2]-b[0]+1)*(b[3]-b[1]+1)
    denom = areaA + areaB - inter + 1e-6
    return inter / denom

class _BTArgs:
    def __init__(self,
                 track_thresh=0.25,
                 match_thresh=0.8,
                 track_buffer=30,
                 frame_rate=30,
                 # Bộ B
                 track_high_thresh=None,
                 track_low_thresh=0.1,
                 new_track_thresh=None,
                 # Khác (một số bản sẽ gọi)
                 fuse_score=True,
                 proximity_thresh=0.5,
                 init_track_thresh=None,
                 mot20=False,
                 min_box_area=0,
                 aspect_ratio_thresh=1.6):
        # Bộ A
        self.track_thresh = float(track_thresh)
        self.match_thresh = float(match_thresh)
        self.track_buffer = int(track_buffer)
        self.frame_rate   = int(frame_rate)
        # Bộ B
        self.track_high_thresh = float(track_high_thresh if track_high_thresh is not None else track_thresh)
        self.track_low_thresh  = float(track_low_thresh)
        self.new_track_thresh  = float(new_track_thresh if new_track_thresh is not None else track_thresh)
        # Các trường bổ sung
        self.fuse_score        = bool(fuse_score)
        self.proximity_thresh  = float(proximity_thresh)
        self.init_track_thresh = float(init_track_thresh if init_track_thresh is not None else self.track_high_thresh)
        # Phụ
        self.mot20 = bool(mot20)
        self.min_box_area = float(min_box_area)
        self.aspect_ratio_thresh = float(aspect_ratio_thresh)

class _Detections:
    def __init__(self, dets):
        if dets is None or len(dets) == 0:
            self.xyxy = np.zeros((0, 4), dtype=float)
            self.conf = np.zeros((0,), dtype=float)
            self.cls  = np.zeros((0,), dtype=float)
        else:
            arr = np.array(dets, dtype=float)
            self.xyxy = arr[:, :4]
            self.conf = arr[:, 4]
            self.cls  = arr[:, 5]
        # tính sẵn xywh
        if self.xyxy.size == 0:
            self.xywh = np.zeros((0, 4), dtype=float)
        else:
            x1y1 = self.xyxy[:, :2]
            x2y2 = self.xyxy[:, 2:4]
            wh   = x2y2 - x1y1
            ctr  = x1y1 + wh / 2.0
            self.xywh = np.concatenate([ctr, wh], axis=1)

    def __len__(self):
        return int(self.xyxy.shape[0])

    def _subset(self, idx):
        out = _Detections(None)
        out.xyxy = self.xyxy[idx].copy()
        out.conf = self.conf[idx].copy()
        out.cls  = self.cls[idx].copy()
        if out.xyxy.size == 0:
            out.xywh = np.zeros((0, 4), dtype=float)
        else:
            x1y1 = out.xyxy[:, :2]
            x2y2 = out.xyxy[:, 2:4]
            wh   = x2y2 - x1y1
            ctr  = x1y1 + wh / 2.0
            out.xywh = np.concatenate([ctr, wh], axis=1)
        return out

    def __getitem__(self, idx):
        return self._subset(idx)

class ByteTrackAdapter:
    def __init__(self, track_thresh=0.25, match_thresh=0.8, track_buffer=30, frame_rate=30,
                 smooth_alpha=None):
        try:
            from ultralytics.trackers.byte_tracker import BYTETracker  # type: ignore
        except Exception as e:
            raise ImportError(f"Ultralytics ByteTrack not available: {e}")
        self._BYTETracker = BYTETracker
        self.args = _BTArgs(track_thresh=track_thresh,
                            match_thresh=match_thresh,
                            track_buffer=track_buffer,
                            frame_rate=frame_rate)
        self.bt = self._BYTETracker(self.args)
        # map id -> (cls, score)
        self._id_cls = {}
        self._id_score = {}
        self._id_det_xyxy = {}
        self._id_det_score = {}
        # EMA smoothing (optional)
        self._bbox_ema = {}
        self._smooth_alpha = None
        if smooth_alpha is not None:
            s = str(smooth_alpha).strip().lower()
            if s not in ("", "0", "none"):
                try:
                    self._smooth_alpha = float(s)
                except Exception:
                    self._smooth_alpha = None  # tắt nếu parse lỗi

    @staticmethod
    def _tlwh_to_tlbr(tlwh):
        x, y, w, h = [float(v) for v in tlwh[:4]]
        return np.array([x, y, x + w, y + h], dtype=float)

    @staticmethod
    def _row_to_xyxy(row):
        r = np.asarray(row, dtype=float)
        if r.size < 4:
            return None
        x1, y1, a3, a4 = r[0], r[1], r[2], r[3]
        xyxy = np.array([x1, y1, a3, a4], dtype=float)
        if a3 <= x1 or a4 <= y1:
            xyxy = np.array([x1, y1, x1 + a3, y1 + a4], dtype=float)
        return xyxy

    def _iter_tracks(self, outputs):
        if outputs is None:
            return
        if isinstance(outputs, np.ndarray):
            for row in outputs:
                row = np.asarray(row)
                if row.size < 5:
                    continue
                xyxy = self._row_to_xyxy(row)
                if xyxy is None:
                    continue
                tid = int(row[4])
                yield {'xyxy': xyxy, 'id': tid}
        else:
            for o in outputs:
                if hasattr(o, 'tlbr'):
                    xyxy = np.asarray(o.tlbr, dtype=float)
                elif hasattr(o, 'tlwh'):
                    xyxy = self._tlwh_to_tlbr(np.asarray(o.tlwh, dtype=float))
                else:
                    continue
                tid = int(getattr(o, 'track_id', getattr(o, 'trackid', -1)))
                yield {'xyxy': xyxy, 'id': tid}

    def _attach_cls_score(self, outputs, dets):
        if dets is None or len(dets) == 0:
            return
        for tr in self._iter_tracks(outputs):
            bb = tr['xyxy']
            best_i, best_cls, best_sc, best_box = 0.0, None, None, None
            for d in dets:
                i = iou(bb, d[:4])
                if i > best_i:
                    best_i   = i
                    best_cls = int(d[5])
                    best_sc  = float(d[4])
                    best_box = d[:4].astype(float)
            if best_cls is not None:
                tid = tr['id']
                best_sc = max(0.0, min(1.0, best_sc))
                self._id_cls[tid]       = best_cls
                self._id_score[tid]     = best_sc
                self._id_det_xyxy[tid]  = best_box
                self._id_det_score[tid] = best_sc

    def _smooth_box(self, tid, xyxy):
        if self._smooth_alpha is None:
            return xyxy
        if tid not in self._bbox_ema:
            self._bbox_ema[tid] = xyxy.astype(float)
            return xyxy
        ema = self._bbox_ema[tid]
        a = self._smooth_alpha
        ema = a * ema + (1.0 - a) * xyxy
        self._bbox_ema[tid] = ema
        return ema

    def update(self, dets, frame_shape=None):
        det_obj = _Detections(dets)
        try:
            outputs = self.bt.update(det_obj)
        except TypeError:
            outputs = self.bt.update(det_obj)

        self._attach_cls_score(outputs, dets)

        out = []
        for tr in self._iter_tracks(outputs):
            xyxy = tr['xyxy']
            tid  = tr['id']

            det_box = self._id_det_xyxy.get(tid, None)
            if det_box is not None and iou(xyxy, det_box) < 0.90:
                xyxy = det_box

            if xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]:
                continue

            xyxy = self._smooth_box(tid, xyxy)

            x1, y1, x2, y2 = xyxy.astype(int).tolist()
            cls = int(self._id_cls.get(tid, -1))
            sc  = float(self._id_det_score.get(tid, self._id_score.get(tid, 0.0)))
            out.append([x1, y1, x2, y2, tid, cls, sc])
        return np.array(out, dtype=object)

# factory
def build_tracker(name: str, **kwargs):
    """
    Đọc tham số từ kwargs, mặc định lấy từ app.config.
    Bản này chỉ hỗ trợ 'bytetrack'.
    """
    name = (name or "").lower()
    if name and name != "bytetrack":
        raise ValueError(f"Only 'bytetrack' is supported in this build. Got: {name}")

    # === defaults từ config ===
    track_thresh = float(kwargs.get("track_thresh", CFG_TRACK_THRESH))
    match_thresh = float(kwargs.get("match_thresh", CFG_MATCH_THRESH))
    track_buffer = int(kwargs.get("track_buffer", CFG_TRACK_BUFFER))
    frame_rate   = int(kwargs.get("frame_rate",   CFG_FRAME_RATE))

    # Bộ B
    track_high_thresh = float(kwargs.get("track_high_thresh",
                                         CFG_TRACK_HIGH_TH if CFG_TRACK_HIGH_TH is not None else track_thresh))
    track_low_thresh  = float(kwargs.get("track_low_thresh",  CFG_TRACK_LOW_TH))
    new_track_thresh  = float(kwargs.get("new_track_thresh",
                                         CFG_NEW_TRACK_TH if CFG_NEW_TRACK_TH is not None else track_thresh))

    # Bổ sung
    fuse_score        = bool(kwargs.get("fuse_score",       CFG_FUSE_SCORE))
    proximity_thresh  = float(kwargs.get("proximity_thresh", CFG_PROXIMITY_TH))
    init_track_thresh = float(kwargs.get("init_track_thresh",
                                         CFG_INIT_TRACK_TH if CFG_INIT_TRACK_TH is not None else track_high_thresh))

    # Smoothing
    smooth_alpha_kw = kwargs.get("smooth_alpha", CFG_SMOOTH_ALPHA)

    adapter = ByteTrackAdapter(
        track_thresh=track_thresh,
        match_thresh=match_thresh,
        track_buffer=track_buffer,
        frame_rate=frame_rate,
        smooth_alpha=smooth_alpha_kw,  # "" / "0" / None -> tắt
    )
    # đảm bảo đầy đủ bộ A/B + extras
    adapter.args.track_high_thresh = track_high_thresh
    adapter.args.track_low_thresh  = track_low_thresh
    adapter.args.new_track_thresh  = new_track_thresh
    adapter.args.fuse_score        = fuse_score
    adapter.args.proximity_thresh  = proximity_thresh
    adapter.args.init_track_thresh = init_track_thresh

    print(f"[tracker] ByteTrack enabled "
          f"(high={adapter.args.track_high_thresh}, low={adapter.args.track_low_thresh}, "
          f"new={adapter.args.new_track_thresh}, match={adapter.args.match_thresh}, "
          f"buffer={adapter.args.track_buffer}, fps={adapter.args.frame_rate}, "
          f"fuse_score={adapter.args.fuse_score})")
    return adapter
