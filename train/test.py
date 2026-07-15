import os
import joblib
import numpy as np
from src.models.hybrid_IDS import Hybrid_IDS
#from evaluate.metric3 import evaluation

def main():
    print("Loading data...")
    X_test_path = "datasets/processed/X_test_path.joblib"
    scaler_path = "datasets/processed/scaler.joblib"
    y_test_path = "datasets/processed/y_test_path.joblib"
    rf_path = "datasets/processed/random_forest_model.joblib"
    iso_forest_path = "datasets/processed/isolation_forest.joblib"

    required_files = [X_test_path, scaler_path, y_test_path, rf_path, iso_forest_path]

    if all(os.path.exists(f) for f in required_files):
        X_test = joblib.load(X_test_path)
        scaler = joblib.load(scaler_path)
        y_test = joblib.load(y_test_path)
        rf = joblib.load(rf_path)
        iso_forest = joblib.load(iso_forest_path)

    else :
        raise ValueError("Error: dataset files not exists.\nMake sure that X_test , y_test, random forest" \
        " and isolation forest exists in your dataset repositary")

    X_test_scaled = scaler.transform(X_test).to_numpy()

    for threshold in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        hybrid = Hybrid_IDS(rf, iso_forest, anomaly_threshold=threshold)
        predicts = hybrid.predict(X_test_scaled)
    
        y_test_arr = y_test.to_numpy()
        rf_predict = rf.predict(X_test_scaled)
    
        true_attack_mask = y_test_arr != 0
        benign_mask = rf_predict == 0
    
        caught = np.sum((predicts == -1) & true_attack_mask)
        false_alarms = np.sum((predicts == -1) & ~true_attack_mask)
    
        print(f"Threshold {threshold}: caught={caught}/19, false_alarms={false_alarms}")

if __name__ == "__main__":
    
    main()