import numpy as np
from sklearn.cluster import KMeans

class KMeansModule:

#K-Means: gom cụm & sinh tín hiệu proxy anomaly (cluster_id, dist_min).

    def __init__(self, n_clusters: int = 5, random_state: int = 42, n_init: str | int = "auto"):
        self.km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)

    def fit(self, X: np.ndarray):
        self.km.fit(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.km.predict(X)

    def distances(self, X: np.ndarray) -> np.ndarray:
        # khoảng cách Euclidean tới mỗi tâm
        # trả về (N x K)
        dists = []
        for row in X:
            d = np.linalg.norm(self.km.cluster_centers_ - row, axis=1)
            dists.append(d)
        return np.stack(dists, axis=0)
