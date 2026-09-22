from typing import Dict, List
import numpy as np

# Bộ feature "thẳng" (đầu vào cho scaler/PCA/KMeans/RF)
BASE_FEATURE_KEYS: List[str] = [
    "dwell",
    "score_mean", "score_std", "score_max",
    "area_mean", "area_std",
    "aspect_mean", "aspect_std",
    "cx_mean", "cy_mean",
    "motion_mean", "motion_peak",
    "blur_mean",
    "iou_stability",
    "mean_speed",
]

# Key dùng để lưu embedding (MobileNetV2)
EMBEDDING_KEY: str = "emb"
# Kích thước embedding MobileNetV2.features global-pooled
EMBEDDING_DIM: int = 1280

# ALL_BASE_KEYS: base features + một khóa 'emb' (emb sẽ mở rộng thành EMBEDDING_DIM giá trị)
ALL_BASE_KEYS: List[str] = BASE_FEATURE_KEYS + [EMBEDDING_KEY]

# Bộ feature cho meta (LR) = p_rf + một vài tín hiệu chính
META_FEATURE_KEYS: List[str] = [
    "dwell", "score_mean", "blur_mean",
    "motion_mean", "iou_stability", "mean_speed"
]


def feats_to_ndarray(feats: Dict[str, float], keys: List[str]) -> np.ndarray:
    """
    Chuyển dict features -> 2D ndarray (1 x tổng số features) theo thứ tự keys.
    Nếu key == EMBEDDING_KEY thì kỳ vọng feats[EMBEDDING_KEY] là list có EMBEDDING_DIM phần tử.
    Nếu không tồn tại hoặc kích thước sai, chèn EMBEDDING_DIM số 0.
    """
    X_row = []
    for k in keys:
        if k == EMBEDDING_KEY:
            val = feats.get(k)
            if isinstance(val, list) and len(val) == EMBEDDING_DIM:
                # mở rộng embedding list
                X_row.extend([float(v) for v in val])
            else:
                X_row.extend([0.0] * EMBEDDING_DIM)
        else:
            v = feats.get(k)
            try:
                X_row.append(float(v) if v is not None else 0.0)
            except Exception:
                # nếu v không phải số (ví dụ string) -> cố gắng chuyển, nếu fail -> 0.0
                try:
                    X_row.append(float(str(v)))
                except Exception:
                    X_row.append(0.0)
    return np.array([X_row], dtype=float)
