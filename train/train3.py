import os
import pandas as pd
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from src.models.hybrid_IDS import Hybrid_IDS, RandomForest, IsolationForest
from evaluate.metric3 import evaluation_curve, evaluation

def tune_threshold(X_test: npt.NDArray[np.float32], y_test: npt.NDArray[np.int16], rf: RandomForest, iso_forest: IsolationForest) :

    precisions, recalls, thresholds = evaluation_curve(X_test, y_test, rf, iso_forest, None)

    for i in range(0, len(thresholds), len(thresholds)//20):
        print(f"Threshold: {thresholds[i]:.4f} | Precision: {precisions[i]:.4f} | recall: {recalls[i]:.4f}")


def visualize_tresholds(precisions: npt.NDArray[np.float32], recalls: npt.NDArray[np.float32], thresholds: npt.NDArray[np.float32], scores: npt.NDArray[np.float32]):
    data = {
        "precision": precisions,
        "recall": recalls 
    }
    df = pd.DataFrame(data)
    
    sns.scatterplot(data=df, x=thresholds, y=scores, hue="threshold", palette="Set2")

    plt.title("Visualize Thresholds") #type: ignore
    plt.show() #type: ignore

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

    hybrid = Hybrid_IDS(rf, iso_forest)

    predicts = hybrid.predict(X_test_scaled)

    Accuracy, classification, confusion = evaluation(y_test.to_numpy(), predicts)

    print(f"Accuracy: {Accuracy}")
    print(f"Classification report: {classification}")
    print(f"Confusion matrix: {confusion}")

    tune_threshold(X_test_scaled, y_test, rf, iso_forest)

    return

if __name__ == "__main__":
    
    main()







