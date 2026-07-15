import numpy as np
from typing import Any
from evaluate.metric_friday import evaluate_family
from src.preprocessing.cleaner import cleaner
from src.models.hybrid_IDS import Hybrid_IDS
import joblib

def load_and_preprocess(csv_path: str, scaler: Any):
    df_clean = cleaner(csv_path)
    y_str = df_clean['Label'].to_numpy()
    X = df_clean.drop(columns=['Label'])

    X_scaled = scaler.transform(X).to_numpy()

    return X_scaled, y_str


def main():
    random_forest = joblib.load("datasets/processed/random_forest_model.joblib")
    isolation_forest = joblib.load("datasets/processed/isolation_forest.joblib")
    Hybrid = Hybrid_IDS(random_forest, isolation_forest, anomaly_threshold=0.447)
    scaler = joblib.load("datasets/processed/scaler.joblib")

    friday_files = {
        "DDos": "datasets/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        "PortScan": "datasets/raw/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        "Botnet": "datasets/raw/Friday-WorkingHours-Morning.pcap_ISCX.csv"
    }

    for family, path in friday_files.items():
        X_scaled, y_str = load_and_preprocess(path, scaler)

        n_attacks = np.sum( y_str != 'BENIGN')
        n_benign = np.sum( y_str == 'BENIGN')

        print(f"\n{family}:")
        print(f"  Shape: {X_scaled.shape}")
        print(f"  Benign: {n_benign}, Attacks: {n_attacks}")
        print(f"  Unique labels: {np.unique(y_str)}")

        evaluate_family(family, X_scaled, y_str, random_forest, isolation_forest, Hybrid)

        

if __name__ == "__main__":
    main()
