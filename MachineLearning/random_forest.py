import numpy as np
from sklearn.ensemble import RandomForestClassifier

class RandomForestModule:
    """
    Bộ phân loại chính: Random Forest.
    """
    def __init__(self, n_estimators: int = 300, max_depth: int | None = None, class_weight="balanced", random_state: int = 42):
        self.rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=-1
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.rf.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.rf.predict_proba(X)

    @property
    def feature_importances_(self):
        return getattr(self.rf, "feature_importances_", None)
