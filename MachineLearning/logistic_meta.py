import numpy as np
from sklearn.linear_model import LogisticRegression

class LogisticMetaModule:
    """
    Meta-model / calibration: Logistic Regression nhận p_rf + một số tín hiệu “thẳng”.
    """
    def __init__(self, C: float = 1.0, penalty: str = "l2", random_state: int = 42):
        self.lr = LogisticRegression(C=C, penalty=penalty, solver="lbfgs", random_state=random_state, max_iter=1000)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.lr.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.lr.predict_proba(X)
