import numpy as np
import numpy.typing as npt
from src.models.random_forest import RandomForest
from src.models.isolation_forest import IsolationForest

class WeightedHybrid_IDS:
    def __init__(self,
                 rf: RandomForest,
                 iforest: IsolationForest,
                 w_rf: float,
                 w_if: float,
                 threshold: float
                ) -> None:
        self.rf = rf
        self.iforest = iforest
        self.w_rf = w_rf
        self.w_if = w_if
        self.threshold = threshold

    def combined_score(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
        rf_proba = 1 - self.rf.predict_proba(X)[:, 0]
        if_score = self.iforest.score(X)
        
        final_score = self.w_rf * rf_proba + self.w_if * if_score

        return final_score
    
    def predict(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.int16]:
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples, dtype=np.int16)

        score = self.combined_score(X)

        for i in range(n_samples):
            predictions[i] = 1 if score >= self.threshold else 0

        return predictions
