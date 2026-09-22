import os
import numpy as np
from typing import Dict, Any

from PIL import Image

# Import keys và converter
from MachineLearning.schema import ALL_BASE_KEYS, META_FEATURE_KEYS, feats_to_ndarray, EMBEDDING_DIM
from MachineLearning.utils_io import load_pickle
# Dùng lại hàm MobileNetV2 từ mobilenetv2_embeddings.py
from MachineLearning.mobilenetv2_embeddings import load_mobilenetv2, embed_pil_image

ART_DIR = os.getenv("ML_ART_DIR", os.path.join(os.path.dirname(__file__), "ml_artifacts"))

# Thư mục chứa snapshots (dùng để resolve đường dẫn tương đối)
SNAPSHOT_DIR = os.getenv(
    "ML_SNAPSHOT_DIR",
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "app", "snapshots"))
)

# Model MobileNetV2 dùng chung cho toàn module (load một lần)
_EMB_MODEL = None


def _get_emb_model(device: str = "cpu"):
    global _EMB_MODEL
    if _EMB_MODEL is None:
        _EMB_MODEL = load_mobilenetv2(device=device)
    return _EMB_MODEL


def _resolve_snapshot_path(path_str: str) -> str | None:
    """
    Resolve đường dẫn ảnh snapshot:
    - Nếu path_str là absolute và tồn tại -> dùng luôn.
    - Nếu là relative -> join với SNAPSHOT_DIR.
    - Nếu không tồn tại -> trả None.
    """
    if not path_str:
        return None
    # absolute path
    if os.path.isabs(path_str) and os.path.exists(path_str):
        return path_str
    # relative -> join base
    candidate = os.path.join(SNAPSHOT_DIR, path_str)
    if os.path.exists(candidate):
        return candidate
    return None


def _choose_snapshot_path(snapshots: Dict[str, Any]) -> str | None:
    """
    Ưu tiên chọn ảnh theo thứ tự:
    1) crop
    2) end
    3) start
    """
    if not snapshots:
        return None
    for key in ("crop", "end", "start"):
        v = snapshots.get(key)
        if not v:
            continue
        p = _resolve_snapshot_path(v)
        if p:
            return p
    return None


def _ensure_embedding(feats_event: Dict[str, Any],
                      snapshots: Dict[str, Any] | None,
                      model,
                      device: str = "cpu") -> Dict[str, Any]:

    emb = feats_event.get("emb")
    if isinstance(emb, list) and len(emb) == EMBEDDING_DIM:
        return feats_event  # đã OK

    if snapshots:
        snap_path = _choose_snapshot_path(snapshots)
        if snap_path:
            try:
                with Image.open(snap_path) as img:
                    img = img.convert("RGB")
                    emb_arr = embed_pil_image(img, model, device=device)
                    feats_event["emb"] = emb_arr.tolist() if hasattr(emb_arr, "tolist") else list(emb_arr)
                    return feats_event
            except Exception as e:
                print(f"[WARN] Failed to compute embedding from {snap_path}: {e}")

    # fallback: vector 0
    feats_event["emb"] = [0.0] * EMBEDDING_DIM
    return feats_event


class RiskModel:
    def __init__(self, device: str = None):
        # device cho MobileNetV2: "cpu" hoặc "cuda"
        self.device = device or os.getenv("ML_DEVICE", "cpu")
        self.scaler = load_pickle(os.path.join(ART_DIR, "scaler.pkl"))
        self.pca = load_pickle(os.path.join(ART_DIR, "pca.pkl"))
        self.kmeans = load_pickle(os.path.join(ART_DIR, "kmeans.pkl"))
        self.rf = load_pickle(os.path.join(ART_DIR, "rf.pkl"))
        self.lr = load_pickle(os.path.join(ART_DIR, "lr_meta.pkl"))

        # load model embedding một lần
        self.emb_model = _get_emb_model(self.device)

    def predict(self, event: Dict[str, Any]) -> Dict[str, Any]:

        feats_event = dict(event.get("features", {}))  # copy để tránh sửa dữ liệu gốc
        snapshots = event.get("snapshots", {})

        # 0) Đảm bảo embedding
        feats_event = _ensure_embedding(feats_event, snapshots, self.emb_model, device=self.device)

        # 1) X_raw bao gồm emb nếu có
        X_raw = feats_to_ndarray(feats_event, ALL_BASE_KEYS)  # shape (1, 14 + EMBEDDING_DIM)

        # 2) Scaler + PCA
        Xs = self.scaler.transform(X_raw) if self.scaler is not None else X_raw
        Xp = self.pca.transform(Xs) if self.pca is not None else Xs

        # 3) KMeans features
        if self.kmeans is not None:
            cid = int(self.kmeans.predict(Xp)[0])
            K = int(self.kmeans.n_clusters)
            km_onehot = np.zeros((1, K), dtype=float)
            km_onehot[0, cid] = 1.0
            ctr = self.kmeans.cluster_centers_[cid]
            dist = float(np.linalg.norm(Xp[0] - ctr))
            dist_min = np.array([[dist]], dtype=float)
        else:
            km_onehot = np.zeros((1, 0), dtype=float)
            cid = None
            dist_min = np.array([[0.0]], dtype=float)

        # 4) Hợp nhất → RF
        X_star = np.concatenate([X_raw, Xp, km_onehot, dist_min], axis=1)
        p_rf = float(self.rf.predict_proba(X_star)[0, 1]) if self.rf is not None else 0.5

        # 5) Meta (LR): p_rf + META_FEATURE_KEYS
        meta_part = feats_to_ndarray(feats_event, META_FEATURE_KEYS)  # shape (1, len(META_FEATURE_KEYS))
        Z = np.concatenate([np.array([[p_rf]], dtype=float), meta_part], axis=1)
        risk = float(self.lr.predict_proba(Z)[0, 1]) if self.lr is not None else min(1.0, max(0.0, 0.4 + 0.6 * p_rf))

        out = {
            "p_rf": p_rf,
            "risk": risk,
            "kmeans": {
                "cluster_id": cid,
                "dist_min": float(dist_min[0, 0]) if dist_min.size else None
            }
        }
        return out
