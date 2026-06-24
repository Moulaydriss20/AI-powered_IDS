import os
import numpy as np
import joblib
from src.models.isolation_forest import IsolationForest
from src.preprocessing.cleaner import cleaner
from src.preprocessing.encoder import Label_encoding
from src.preprocessing.spliter import spliter
from evaluate.metrics2 import evaluation
from evaluate.metric3 import evaluation_curve

def main():
    print("Loading data...")
    X_train_path = 'datasets/processed/X_train_path.joblib'
    y_train_path = 'datasets/processed/y_train_path.joblib'
    rf_path = "datasets/processed/random_forest_model.joblib"
    scaler_path = 'datasets/processed/scaler.joblib'

    required_files = [X_train_path, y_train_path, scaler_path]

    if all(os.path.exists(f) for f in required_files):
        X_train = joblib.load(X_train_path)
        y_train = joblib.load(y_train_path)
        rf = joblib.load(rf_path)
        scaler = joblib.load(scaler_path)

    else:
        X_train , _, y_train, _ = cleaner('datasets/raw/Wednesday-workingHours.pcap_ISCX.csv').pipe(Label_encoding).pipe(spliter)
        scaler = joblib.load(scaler_path)
        rf = joblib.load(rf_path)
        

    X_train_scaled = scaler.transform(X_train).to_numpy()

    forest = IsolationForest()
    print("--- Start Training ---")
    forest.fit(X_train_scaled)

# The anomaly should have a shorter path than normal points
    print("--- Start score computation ---")
    scores = forest.score(X_train_scaled)

    print("--- Start prediction ---")
    predicts = forest.predict(X_train_scaled, scores)
    print("--- Prediction ended ---")

    print(f"scores : \n{scores}")

    print(f"Predictions : \n{predicts}")

    y_samples = y_train.to_numpy()

    benign_scores = scores[y_samples == 0]
    attack_scores = scores[y_samples != 0]

    print(f"Average BENIGN score: {benign_scores.mean():.4f}")
    print(f"Average ATTACK score: {attack_scores.mean():.4f}")

    print(f"Predicted anomalies: {np.sum(predicts == -1)}")
    print(f"Actual attacks in sample: {np.sum(y_samples != 0)}")

    y_true_binary = np.where(y_samples == 0, 1, -1)

    classification = evaluation(y_true_binary, predicts)

    print(f"Classification report :\n{classification}")

    precisions, recalls, thresholds = evaluation_curve(X_train_scaled, y_samples, rf, forest, None)

    for i in range(0, len(thresholds), len(thresholds)//20):
        print(f"Threshold: {thresholds[i]:.4f} | Precision: {precisions[i]:.4f} | recall: {recalls[i]:.4f}")

    joblib.dump(forest, "datasets/processed/isolation_forest.joblib")

if __name__ == "__main__":
   main()