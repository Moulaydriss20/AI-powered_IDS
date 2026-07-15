import os
import pandas as pd
import numpy as np
import numpy.typing as npt
import joblib
import matplotlib.pyplot as plt
from src.preprocessing.cleaner import cleaner
from src.preprocessing.encoder import Label_encoding
from src.preprocessing.spliter import spliter

def compute_pr_curve(scores: npt.NDArray[np.float32],
                     y_true: npt.NDArray[np.int32]
                    ) -> tuple[
                        npt.NDArray[np.float32],
                        npt.NDArray[np.float32],
                        npt.NDArray[np.float32]
                    ]:
    
    thresholds = np.unique(scores)
    precicions = np.zeros(len(thresholds), dtype=np.float32)
    recalls = np.zeros(len(thresholds), dtype=np.float32)

    for i, threshold in enumerate(thresholds):

        predicted_attack = (scores >= threshold)

        TP = np.sum(predicted_attack & (y_true == 1))
        FP = np.sum(predicted_attack & (y_true == 0))
        FN = np.sum(~predicted_attack & (y_true == 1))

        precicions[i] = TP / (TP + FP)
        
        recalls[i] = TP / (TP + FN)


    return thresholds, precicions, recalls

def find_best_f1_threshold(thresholds: npt.NDArray[np.float32],
                            presicions: npt.NDArray[np.float32],
                            recalls: npt.NDArray[np.float32]
                           ) -> tuple[float, float, float, float]:
    
    denom = presicions + recalls
    F1 = np.where(denom > 0, 2 * presicions * recalls / denom, 0.0)

    best_threshold = 0
    best_f1 = 0
    presicions_at_best = 0
    recall_at_best = 0

    for i, threshold in enumerate(thresholds):
        if F1[i] >= best_f1:
            best_threshold = threshold
            best_f1 = F1[i]
            presicions_at_best = presicions[i]
            recall_at_best = recalls[i]

    return best_threshold, best_f1, presicions_at_best, recall_at_best

def report_at_target(target: float, thresholds: npt.NDArray[np.float32], precisions: npt.NDArray[np.float32], recalls: npt.NDArray[np.float32], label: str):
    """Find nearest threshold to target and report metrics."""
    idx = np.argmin(np.abs(thresholds - target))
    actual_threshold = thresholds[idx]
    print(f"{label} (target={target:.2f}, actual={actual_threshold:.4f}) | "
          f"Precision: {precisions[idx]:.4f} | Recall: {recalls[idx]:.4f}")

def main():
    print("Loading data...")
    X_test_path = 'datasets/processed/X_test_path.joblib'
    y_test_path = 'datasets/processed/y_test_path.joblib'
    scaler_path = 'datasets/processed/scaler.joblib'

    required_files = [X_test_path, y_test_path, scaler_path]

    if all(os.path.exists(f) for f in required_files):
        X_test = joblib.load(X_test_path)
        y_test = joblib.load(y_test_path)
        scaler = joblib.load(scaler_path)

    else:
        _ , X_test, y_test, _ = cleaner('datasets/raw/Wednesday-workingHours.pcap_ISCX.csv').pipe(Label_encoding).pipe(spliter)

        scaler = joblib.load(scaler_path)

    X_test_scaled = scaler.transform(X_test).to_numpy()

    isf = joblib.load('datasets/processed/isolation_forest.joblib')

    scores = isf.score(X_test_scaled)

    y_true = np.where(y_test != 0, 1, 0)

    thresholds, precisions, recalls = compute_pr_curve(scores, y_true)

    best_threshold, best_f1, precision_at_best, recall_at_best = find_best_f1_threshold(thresholds, precisions, recalls)

    report_at_target(0.5, thresholds, precisions, recalls, "Baseline (Liu et al.)")
    report_at_target(0.77, thresholds, precisions, recalls, "Hybrid-coherent")

    print(f"Threshold: {best_threshold:.4f}| F1 score: {best_f1:.4f} | Precision: {precision_at_best:.4f} | recall: {recall_at_best:.4f}")

    plt.figure(figsize=(8, 6)) #type: ignore
    plt.plot(recalls, precisions, linewidth=2, color='#2E5F8A') #type: ignore
    for target, label, color in [(0.5, 'Baseline (0.5)', 'green'),
                              (0.77, 'Hybrid-coherent (0.77)', 'red'),
                              (best_threshold, f'Best F1 ({best_threshold:.3f})', 'purple')]:
        idx = np.argmin(np.abs(thresholds - target))
        plt.scatter(recalls[idx], precisions[idx], s=100, color=color, #type: ignore 
                label=label, zorder=5, edgecolors='black')

    plt.xlabel('Recall (attack detection rate)', fontsize=12) #type: ignore
    plt.ylabel('Precision', fontsize=12) #type: ignore
    plt.title('Isolation Forest — Standalone PR Curve on Wednesday Test Set', fontsize=13) #type: ignore
    plt.legend(loc='best') #type: ignore
    plt.grid(True, alpha=0.3) #type: ignore
    plt.tight_layout()
    plt.savefig('if_standalone_pr_curve.png', dpi=150) #type: ignore
    print("PR curve saved if_standalone_pr_curve.png")

    results_df = pd.DataFrame({
        'Operating Point': ['Baseline (0.5)', 'Hybrid-coherent (0.77)', f'Best F1 ({best_threshold:.4f})'],
        'Threshold': [thresholds[np.argmin(np.abs(thresholds - 0.5))],
                        thresholds[np.argmin(np.abs(thresholds - 0.77))],
                        best_threshold],
        'Precision': [precisions[np.argmin(np.abs(thresholds - 0.5))],
                        precisions[np.argmin(np.abs(thresholds - 0.77))],
                        precision_at_best],
        'Recall': [recalls[np.argmin(np.abs(thresholds - 0.5))],
                   recalls[np.argmin(np.abs(thresholds - 0.77))],
                   recall_at_best],
        'F1': [2 * precisions[np.argmin(np.abs(thresholds - 0.5))] * recalls[np.argmin(np.abs(thresholds - 0.5))] / (precisions[np.argmin(np.abs(thresholds - 0.5))] + recalls[np.argmin(np.abs(thresholds - 0.5))] + 1e-10),
                2 * precisions[np.argmin(np.abs(thresholds - 0.77))] * recalls[np.argmin(np.abs(thresholds - 0.77))] / (precisions[np.argmin(np.abs(thresholds - 0.77))] + recalls[np.argmin(np.abs(thresholds - 0.77))] + 1e-10),
                best_f1],
    })

    results_df.to_csv('if_standalone_results.csv', index=False)
    print(f"\nResults table:\n{results_df.to_string(index=False)}")

if __name__ == "__main__":
    main()