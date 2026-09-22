import os
import json
import numpy as np
from typing import Dict, List

from sklearn.preprocessing import StandardScaler
from MachineLearning.schema import ALL_BASE_KEYS, META_FEATURE_KEYS, feats_to_ndarray, EMBEDDING_KEY, EMBEDDING_DIM
from MachineLearning.pca import PCAModule
from MachineLearning.kmeans import KMeansModule
from MachineLearning.random_forest import RandomForestModule
from MachineLearning.logistic_meta import LogisticMetaModule
from MachineLearning.utils_io import ensure_dir, save_pickle

DEFAULT_TRAIN_JSON = os.path.join(
    os.path.dirname(__file__),
    "tools",
    "train_with_emb.json"
)

ART_DIR = os.getenv("ML_ART_DIR", os.path.join(os.path.dirname(__file__), "ml_artifacts"))

def _stack_rows(dict_rows: List[Dict[str, float]], keys: List[str]) -> np.ndarray:
    X = []
    for row in dict_rows:
        X_row = feats_to_ndarray(row, keys)[0]
        X.append(X_row)
    return np.array(X, dtype=float)

def train_pipeline(train_json_path: str, n_pca: int = 10, k_clusters: int = 5):
    print(f"[INFO] Loading train JSON: {train_json_path}")

    # 1) Load dữ liệu
    with open(train_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    feats = [d["features"] for d in data]
    y = np.array([int(d["label"]) for d in data], dtype=int)

    # 2) X_raw theo ALL_BASE_KEYS (BASE_FEATURE_KEYS + emb)
    X_raw = _stack_rows(feats, ALL_BASE_KEYS)  # shape (n, 14 + EMBEDDING_DIM)

    # 3) Scaler + PCA
    scaler = StandardScaler().fit(X_raw)
    Xs = scaler.transform(X_raw)
    pca = PCAModule(n_components=n_pca).fit(Xs)
    Xp = pca.transform(Xs)

    # 4) KMeans + one-hot + dist_min
    km = KMeansModule(n_clusters=k_clusters).fit(Xp)
    cids = km.predict(Xp)
    K = km.km.n_clusters
    onehot = np.zeros((len(cids), K), dtype=float)
    onehot[np.arange(len(cids)), cids] = 1.0
    dists = km.distances(Xp)
    dist_min = dists.min(axis=1).reshape(-1, 1)

    # 5) Hợp nhất đặc trưng cho RF
    # X_star = [X_raw, Xp, onehot, dist_min]
    X_star = np.concatenate([X_raw, Xp, onehot, dist_min], axis=1)

    # 6) Train RF
    rf = RandomForestModule().fit(X_star, y)

    # 7) Meta features + train LR (meta/calibration)
    # p_rf từ RF + META_FEATURE_KEYS
    p_rf = rf.predict_proba(X_star)[:, 1].reshape(-1, 1)

    # Lấy đúng các meta features (n x len(META_FEATURE_KEYS))
    meta_raw = _stack_rows(feats, META_FEATURE_KEYS)  # returns n x 6
    Z_meta = np.concatenate([p_rf, meta_raw], axis=1)
    lr = LogisticMetaModule().fit(Z_meta, y)

    # 8) Lưu artifacts
    ensure_dir(ART_DIR)
    save_pickle(scaler, os.path.join(ART_DIR, "scaler.pkl"))
    save_pickle(pca.pca, os.path.join(ART_DIR, "pca.pkl"))
    # lưu sklearn KMeans object
    save_pickle(km.km, os.path.join(ART_DIR, "kmeans.pkl"))
    save_pickle(rf.rf, os.path.join(ART_DIR, "rf.pkl"))
    save_pickle(lr.lr, os.path.join(ART_DIR, "lr_meta.pkl"))

    print(f"[OK] Saved artifacts to: {ART_DIR}")
    print(f"PCA n_components={pca.n_components_}")
    print(f"KMeans K={K}")

if __name__ == "__main__":
    train_pipeline(DEFAULT_TRAIN_JSON, n_pca=10, k_clusters=5)
