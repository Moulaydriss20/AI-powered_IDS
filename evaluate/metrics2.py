import numpy as np
import numpy.typing as npt
from sklearn.metrics import classification_report #type: ignore
from typing import cast

def evaluation(y: npt.NDArray[np.int16], predicts: npt.NDArray[np.int16]):

    return cast(str, classification_report(y, predicts))

