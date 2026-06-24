import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from typing import cast

def feature_scaling(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame] :
    scaler = StandardScaler()

    scaler.set_output(transform='pandas')

    scaler.fit(X_train)

    X_train_scaled = cast(pd.DataFrame, scaler.transform(X_train))

    X_test_scaled = cast(pd.DataFrame, scaler.transform(X_test))

    joblib.dump(scaler,"D:/development/AI-powered_IDS/datasets/processed/scaler.joblib")

    return X_train_scaled, X_test_scaled