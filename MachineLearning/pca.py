from typing import Optional
import numpy as np
from sklearn.decomposition import PCA

class PCAModule:
    """
    PCA: giảm chiều/khử tương quan & sinh PC-features.
    """
    def __init__(self, n_components: Optional[int] = None, svd_solver: str = "auto", random_state: int = 42):
        self.pca = PCA(n_components=n_components, svd_solver=svd_solver, random_state=random_state)

    def fit(self, X: np.ndarray):
        self.pca.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.pca.transform(X)

    @property
    def explained_variance_ratio_(self):
        return getattr(self.pca, "explained_variance_ratio_", None)

    @property
    def n_components_(self):
        return getattr(self.pca, "n_components_", None)
