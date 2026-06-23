import pandas as pd
from sklearn.model_selection import train_test_split # type: ignore
import joblib
from typing import cast

def spliter(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X, y = df.drop(columns=['Label']), df['Label']

    X_train, X_test, y_train, y_test = cast(tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series], train_test_split(X, y, test_size=0.2, random_state=42 , stratify=y))

    joblib.dump(X_train, "D:/development/AI-powered_IDS/datasets/processed/X_train_path.joblib")
    joblib.dump(X_test, "D:/development/AI-powered_IDS/datasets/processed/X_test_path.joblib")
    joblib.dump(y_train, "D:/development/AI-powered_IDS/datasets/processed/y_train_path.joblib")
    joblib.dump(y_test, "D:/development/AI-powered_IDS/datasets/processed/y_test_path.joblib")

    return X_train, X_test, y_train, y_test