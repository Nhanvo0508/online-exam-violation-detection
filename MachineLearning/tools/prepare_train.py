#!/usr/bin/env python3
"""
prepare_train_with_crops.py

- Read cv_events.jsonl (EVENTS_PATH)
- For each event:
    - prefer snapshots.crop if exists and file found
    - else prefer snapshots.start (fallback snapshots.end)
    - if bbox exists in rec['bbox'] use it (x1,y1,x2,y2)
    - else compute bbox from cx_mean, cy_mean, area_mean, aspect_mean (as in compute_bbox_from_stats)
    - crop and save to OUT_CROP_DIR/{event_id}-crop.jpg
    - set rec['snapshots']['crop'] = absolute_path_to_crop
- Write updated events to CV_EVENTS_WITH_CROPS (jsonl)
- Create train.json (list) at TRAIN_OUT with {"features":..., "label":0/1}
"""

import os
import json
import math
import sys
from PIL import Image

# ====== CONFIG - chỉnh nếu cần ======
EVENTS_PATH = r"C:\Users\anbin\PycharmProjects\TGMT-01\app\cv_events.jsonl"
SNAPSHOT_DIR = r"C:\Users\anbin\PycharmProjects\TGMT-01\app\snapshots"   # nơi snapshot gốc
OUT_CROP_DIR = r"C:\Users\anbin\PycharmProjects\TGMT-01\app\snapshots\crops"
CV_EVENTS_WITH_CROPS = r"C:\Users\anbin\PycharmProjects\TGMT-01\MachineLearning\tools\cv_events_with_crops.jsonl"
TRAIN_OUT = r"C:\Users\anbin\PycharmProjects\TGMT-01\MachineLearning\tools\train.json"

VIOLATION_CLASSES = {"book", "calculator", "cheat_sheet", "earphone", "notebook", "phone"}
# =====================================

os.makedirs(OUT_CROP_DIR, exist_ok=True)
os.makedirs(os.path.dirname(CV_EVENTS_WITH_CROPS), exist_ok=True)
os.makedirs(os.path.dirname(TRAIN_OUT), exist_ok=True)

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for ln_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                print(f"[WARN] skip line {ln_no}: JSON parse error: {e}", file=sys.stderr)
                continue

def normalize_clsname(cn: str) -> str:
    if not cn:
        return ""
    return cn.strip().lower()

def compute_bbox_from_stats(img_w, img_h, cx, cy, area, aspect):
    """
    cx,cy assumed normalized [0,1] (as in your sample)
    area if <=1 assumed normalized fraction of image area else pixel area.
    aspect = width / height
    returns (x1,y1,x2,y2) clipped to image
    """
    # center pixels
    if cx > 1.5 or cy > 1.5:
        cx_pix = float(cx)
        cy_pix = float(cy)
    else:
        cx_pix = float(cx) * img_w
        cy_pix = float(cy) * img_h

    img_area = img_w * img_h
    # interpret area
    try:
        area_f = float(area)
    except:
        area_f = 0.0

    if area_f <= 1.0:
        area_pix = area_f * img_area
    else:
        area_pix = area_f
        if area_pix > img_area * 0.95:
            area_pix = img_area * 0.7

    if aspect is None or aspect <= 0:
        aspect = 1.0
    # compute width/height
    w = math.sqrt(max(area_pix * aspect, 1.0))
    h = max(area_pix / w, 1.0)

    x1 = int(round(cx_pix - w/2.0))
    y1 = int(round(cy_pix - h/2.0))
    x2 = int(round(cx_pix + w/2.0))
    y2 = int(round(cy_pix + h/2.0))

    x1 = max(0, min(x1, img_w-1))
    x2 = max(0, min(x2, img_w-1))
    y1 = max(0, min(y1, img_h-1))
    y2 = max(0, min(y2, img_h-1))

    if x2 <= x1:
        x2 = min(img_w-1, x1 + 10)
    if y2 <= y1:
        y2 = min(img_h-1, y1 + 10)

    return x1, y1, x2, y2

def save_crop_from_snapshot(snapshot_fname, bbox, out_path):
    snap_path = os.path.join(SNAPSHOT_DIR, snapshot_fname)
    if not os.path.exists(snap_path):
        raise FileNotFoundError(f"snapshot not found: {snap_path}")
    with Image.open(snap_path) as img:
        img = img.convert("RGB")
        img_w, img_h = img.size
        x1,y1,x2,y2 = bbox
        # ensure bbox safe
        x1 = max(0, min(x1, img_w-1))
        x2 = max(0, min(x2, img_w-1))
        y1 = max(0, min(y1, img_h-1))
        y2 = max(0, min(y2, img_h-1))
        if x2 <= x1 or y2 <= y1:
            # fallback to small central crop
            cx = img_w//2; cy = img_h//2
            x1 = max(0, cx-20); x2 = min(img_w-1, cx+20)
            y1 = max(0, cy-20); y2 = min(img_h-1, cy+20)
        crop = img.crop((x1,y1,x2,y2))
        crop.save(out_path, quality=85)

def main():
    total = 0
    out_records = []
    n_skipped = 0
    n_cropped = 0
    n_existing = 0

    for rec in load_jsonl(EVENTS_PATH):
        total += 1
        feats = rec.get("features")
        if feats is None:
            n_skipped += 1
            continue

        snaps = rec.get("snapshots", {})
        crop_path = snaps.get("crop")
        crop_ok = False
        if crop_path:
            # if path is relative, try to resolve; if absolute, check
            if not os.path.isabs(crop_path):
                candidate = os.path.join(os.path.dirname(EVENTS_PATH), crop_path)
            else:
                candidate = crop_path
            if os.path.exists(candidate):
                # ensure we store absolute path for consistency
                rec.setdefault("snapshots", {})["crop"] = os.path.abspath(candidate)
                crop_ok = True
                n_existing += 1
        if not crop_ok:
            # need to create crop
            snap_name = snaps.get("start") or snaps.get("end")
            if not snap_name:
                # nothing to crop
                # write record as-is
                out_records.append(rec)
                continue
            # open snapshot to know size etc
            snap_path = os.path.join(SNAPSHOT_DIR, snap_name)
            if not os.path.exists(snap_path):
                print(f"[WARN] snapshot missing for event {rec.get('event_id')} -> {snap_path}", file=sys.stderr)
                out_records.append(rec)
                continue
            # load image to get dimensions
            try:
                with Image.open(snap_path) as img:
                    img = img.convert("RGB")
                    img_w, img_h = img.size
            except Exception as e:
                print(f"[WARN] cannot open snapshot {snap_path}: {e}", file=sys.stderr)
                out_records.append(rec)
                continue

            # determine bbox: prefer explicit rec['bbox'] if present (x1,y1,x2,y2)
            bbox = None
            if "bbox" in rec:
                try:
                    b = rec["bbox"]
                    bbox = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
                except Exception:
                    bbox = None

            if bbox is None:
                # compute from features (cx_mean, cy_mean, area_mean, aspect_mean)
                f = feats
                cx = float(f.get("cx_mean", 0.5))
                cy = float(f.get("cy_mean", 0.5))
                area = float(f.get("area_mean", 0.0) or 0.0)
                aspect = float(f.get("aspect_mean", 1.0) or 1.0)
                bbox = compute_bbox_from_stats(img_w, img_h, cx, cy, area, aspect)

            # save crop
            eid = rec.get("event_id") or f"evt_{total}"
            out_name = f"{eid}-crop.jpg"
            out_path = os.path.join(OUT_CROP_DIR, out_name)
            try:
                save_crop_from_snapshot(snap_name, bbox, out_path)
                rec.setdefault("snapshots", {})["crop"] = os.path.abspath(out_path)
                n_cropped += 1
            except Exception as e:
                print(f"[WARN] failed to crop for event {eid}: {e}", file=sys.stderr)
                # still append record without crop
                pass

        out_records.append(rec)

    # write updated JSONL with crop paths
    with open(CV_EVENTS_WITH_CROPS, "w", encoding="utf-8") as fo:
        for r in out_records:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")

    # build train.json array (features + label)
    train_list = []
    for rec in out_records:
        feats = rec.get("features")
        if feats is None:
            continue
        cn = normalize_clsname(rec.get("cls_name", ""))
        label = 1 if cn in VIOLATION_CLASSES else 0
        train_list.append({"features": feats, "label": label})

    with open(TRAIN_OUT, "w", encoding="utf-8") as fo:
        json.dump(train_list, fo, indent=2, ensure_ascii=False)

    print(f"[OK] processed {total} events")
    print(f"  cropped: {n_cropped}, existing crops: {n_existing}, skipped: {n_skipped}")
    print(f"  wrote events with crops: {CV_EVENTS_WITH_CROPS}")
    print(f"  wrote train.json: {TRAIN_OUT}")

if __name__ == "__main__":
    main()
