import pandas as pd
import numpy as np
import numpy.typing as npt
from src.models.random_forest import RandomForest

def get_feature_importances(rf: RandomForest, X_train: pd.DataFrame) -> list[tuple[str, npt.NDArray[np.float32]]]:

    importances = rf.get_feature_importances(n_features=X_train.shape[1])

    features_names = X_train.columns.tolist()
    importances_df = sorted(zip(features_names, importances), key=lambda x: x[1] , reverse= True)

    return importances_df
