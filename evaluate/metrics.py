import numpy as np
import numpy.typing as npt
from sklearn.metrics import classification_report, confusion_matrix #type: ignore
from typing import cast

def evaluation(y: npt.NDArray[np.int32], preds: npt.NDArray[np.int16]):

    accuracy = cast(float, np.mean(preds == y))
    classification = cast(str, classification_report(y, preds))
    confusion = confusion_matrix(y, preds)

    return accuracy, classification, confusion