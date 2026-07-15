import numpy as np
import numpy.typing as npt
from src.models.random_forest import RandomForest
from src.models.isolation_forest import IsolationForest

class Hybrid_IDS():
    def __init__(self, rf: RandomForest, iso_forest: IsolationForest, anomaly_threshold: float=0.33) -> None:
        self.rf = rf
        self.iso_forest = iso_forest
        self.anomaly_threshold = anomaly_threshold

    def predict(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.int16] :

        rf_predict = self.rf.predict(X)
        iso_forest_score = self.iso_forest.score(X)

        final_predict = rf_predict.copy()

        benign_mask = rf_predict == 0
        anomaly_mask = iso_forest_score >= self.anomaly_threshold

        final_predict[benign_mask & anomaly_mask] = -1

        print("Number of -1 flags:", np.sum(final_predict == -1))
        print("Number of BENIGN-then-anomalous rows:", np.sum(benign_mask & anomaly_mask))
        print("Max IF score:", np.max(iso_forest_score))
        print("Min IF score:", np.min(iso_forest_score))

        return final_predict
    
    def get_predictions(self, X:npt.NDArray[np.float32]) -> tuple[npt.NDArray[np.int16], npt.NDArray[np.float32]]:

        return self.rf.predict(X), self.iso_forest.score(X)
