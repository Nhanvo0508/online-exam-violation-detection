#!/usr/bin/env python3
r"""
mobilenetv2_embeddings.py

Compute MobileNetV2 embeddings for event snapshots.

Usage examples:

# Batch mode - compute embeddings for all events and write train_with_emb.json
python MachineLearning/mobilenetv2_embeddings.py --mode batch \
    --events "MachineLearning/tools/cv_events_with_crops.jsonl" \
    --snapshots-dir "snapshots" \
    --out "MachineLearning/tools/train_with_emb.json" \
    --device cpu
# Single image test (inference helper)
python mobilenetv2_embeddings.py --mode single --img "C:\path\to\crop.jpg" --device cpu

This script:
- Loads MobileNetV2.features (pretrained)
- Global-average-pools the last feature map -> embedding (1280 dims)
- For batch mode: reads events jsonl, locates snapshot crop/start/end, computes emb and writes out JSON list with features['emb']=list(...)
"""

import os
import json
import argparse
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.models as models
import sys

# Default paths (feel free to override with CLI)
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_EVENTS = PROJECT_ROOT / "tools" / "cv_events_with_crops.jsonl"
DEFAULT_SNAP_DIR = PROJECT_ROOT.parent / "snapshots"
DEFAULT_OUT = PROJECT_ROOT / "tools" / "train_with_emb.json"

# Build model once
def load_mobilenetv2(device="cpu"):
    model = models.mobilenet_v2(pretrained=True).features  # excludes classifier
    model.eval()
    model.to(device)
    return model

# transform
TF = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

def embed_pil_image(img_pil: Image.Image, model, device="cpu"):
    """
    img_pil: PIL Image RGB
    model: mobilenet_v2.features module
    returns: 1D numpy array (embedding)
    """
    x = TF(img_pil).unsqueeze(0).to(device)  # 1x3x224x224
    with torch.no_grad():
        feat_map = model(x)  # shape (1, C, H, W)
        emb = feat_map.mean(dim=[2,3]).squeeze(0).cpu().numpy()  # C dims
    return emb.astype(float)

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

def find_snapshot_path(rec, snapshots_dir):
    snaps = rec.get("snapshots", {})
    # prefer 'crop' (already created), else prefer 'start', else 'end'
    for key in ("crop", "start", "end"):
        v = snaps.get(key)
        if not v:
            continue
        # if absolute path, use directly; else join with snapshots_dir
        if os.path.isabs(v) and os.path.exists(v):
            return v
        candidate = os.path.join(snapshots_dir, v)
        if os.path.exists(candidate):
            return candidate
    return None

def process_events_add_embeddings(events_jsonl, snapshots_dir, out_json, device="cpu", model=None, prefer_label_from_rec=True):
    """
    Read events JSONL, compute embedding for chosen snapshot per event,
    attach features['emb'] as list, and write out a JSON list suitable for pipeline_train.py.
    """
    if model is None:
        model = load_mobilenetv2(device=device)

    out_list = []
    n = 0
    n_emb = 0
    n_no_snap = 0
    for rec in load_jsonl(events_jsonl):
        n += 1
        feats = rec.get("features")
        if feats is None:
            continue
        snap_path = find_snapshot_path(rec, snapshots_dir)
        emb = None
        if snap_path:
            try:
                with Image.open(snap_path) as img:
                    img = img.convert("RGB")
                    emb = embed_pil_image(img, model, device=device)
                    n_emb += 1
            except Exception as e:
                print(f"[WARN] failed embed for {snap_path}: {e}", file=sys.stderr)
                emb = None
        else:
            n_no_snap += 1
        # attach embedding (if None, fill zeros of model channel count)
        if emb is None:
            # determine channel dim (MobileNetV2 default 1280)
            emb = np.zeros(1280, dtype=float)
        feats_copy = dict(feats)
        feats_copy["emb"] = emb.tolist()
        # determine label: prefer rec['label'] if exists; else if we want to auto-label use cls_name check outside
        out_rec = {"features": feats_copy}
        if prefer_label_from_rec and "label" in rec:
            out_rec["label"] = int(rec["label"])
        else:
            # leave label absent; downstream prepare_train will set label based on cls_name
            # but pipeline_train requires label - so caller should ensure labels exist (or we can attempt to auto)
            pass
        out_list.append(out_rec)
        if n % 100 == 0:
            print(f"[INFO] processed {n} events, embeddings computed: {n_emb}")

    # If out_json should be list for pipeline_train, ensure every item has label.
    # If some items lack label, you may want to auto-fill using cls_name->VIOLATION_CLASSES mapping.
    # Here we will auto-fill labels by cls_name if missing:
    for item_idx, item in enumerate(out_list):
        if "label" not in item:
            # try to fetch original rec? we didn't keep original rec here.
            # For safety, set label = 1 (user can override with labels.csv later).
            item["label"] = 1

    # write out as JSON list
    with open(out_json, "w", encoding="utf-8") as fo:
        json.dump(out_list, fo, indent=2, ensure_ascii=False)
    print(f"[OK] wrote {len(out_list)} items with embeddings to {out_json} (embeddings computed: {n_emb}, no snapshot: {n_no_snap})")

def embed_single_image(img_path, device="cpu"):
    model = load_mobilenetv2(device=device)
    if not os.path.exists(img_path):
        raise FileNotFoundError(img_path)
    with Image.open(img_path) as img:
        img = img.convert("RGB")
        emb = embed_pil_image(img, model, device=device)
    return emb

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["batch","single"], default="batch", help="batch: process events jsonl; single: embed one image")
    p.add_argument("--events", default=DEFAULT_EVENTS, help="input events jsonl (for batch)")
    p.add_argument("--snapshots-dir", default=DEFAULT_SNAP_DIR, help="snapshots dir")
    p.add_argument("--out", default=DEFAULT_OUT, help="output train json (with embeddings)")
    p.add_argument("--img", default=None, help="single image path for mode=single")
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    args = p.parse_args()

    if args.mode == "single":
        if not args.img:
            print("Please provide --img for single mode", file=sys.stderr)
            return
        emb = embed_single_image(args.img, device=args.device)
        print(f"embedding shape: {emb.shape}")
        # save as npy next to img
        out_npy = os.path.splitext(args.img)[0] + ".npy"
        np.save(out_npy, emb)
        print(f"Saved embedding to {out_npy}")
    else:
        process_events_add_embeddings(args.events, args.snapshots_dir, args.out, device=args.device)

if __name__ == "__main__":
    main()