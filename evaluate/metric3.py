import numpy as np
import numpy.typing as npt
from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix #type: ignore
from src.models.random_forest import RandomForest
from src.models.isolation_forest import IsolationForest
from src.models.hybrid_IDS import Hybrid_IDS
from typing import overload, cast

@overload
def evaluation_curve(X: npt.NDArray[np.float32], y: npt.NDArray[np.int16], rf: RandomForest, iso_forest: IsolationForest, hybrid: None) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]: ...

@overload
def evaluation_curve(X: npt.NDArray[np.float32], y: npt.NDArray[np.int16], rf: None, iso_forest: None, hybrid: Hybrid_IDS) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]: ...


def evaluation_curve(X: npt.NDArray[np.float32], y:npt.NDArray[np.int16], rf: RandomForest | None, iso_forest: IsolationForest | None, hybrid: Hybrid_IDS | None) :
    if rf is not None and iso_forest is not None:
        rf_predict = rf.predict(X)
        iso_forest_score = iso_forest.score(X)

    elif hybrid is not None:    
    
        rf_predict, iso_forest_score =  hybrid.get_predictions(X)

    else : raise ValueError("Error : You must provide rf and iso_forest or an hybrid model")

    mask = rf_predict == 0

    y_true_subset = y[mask]
    iso_score_subset = iso_forest_score[mask]

    ground_truth_binary = np.where(y_true_subset != 0, 1, 0)

    _precision_recall_curve = precision_recall_curve(ground_truth_binary, iso_score_subset)

    print("Total RF-BENIGN rows in test set:", np.sum(mask))
    print("Of those, true attacks hiding inside:", np.sum(y_true_subset != 0))
    print("True attack breakdown by class:", np.unique(y_true_subset[y_true_subset != 0], return_counts=True))  

    return _precision_recall_curve

def evaluation(y: npt.NDArray[np.int16], predictions: npt.NDArray[np.int16]) -> tuple[float, str, npt.NDArray[np.float32]]:
    accuracy = np.mean(predictions == y, dtype=float)
    classification = cast(str, classification_report(y, predictions))
    confusion = confusion_matrix(y, predictions)

    return accuracy , classification, confusion