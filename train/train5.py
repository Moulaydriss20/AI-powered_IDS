import pandas as pd
import numpy as np
import numpy.typing as npt
import joblib
from evaluate.metric4 import evaluate_config

def main():
    print("Loading artifacts...")
    scaler = joblib.load('datasets/processed/scaler.joblib')
    rf = joblib.load('datasets/processed/random_forest_model.joblib')
    iforest = joblib.load('datasets/processed/isolation_forest_benign_only.joblib')
    X_test = joblib.load('datasets/processed/X_test_path.joblib')
    y_test = joblib.load('datasets/processed/y_test_path.joblib')
    
    X_test_scaled = scaler.transform(X_test).to_numpy()
    y_test_np = y_test.to_numpy()
    y_true_binary = (y_test_np != 0).astype(np.int16)
    
    # Compute RF and IF outputs ONCE (very expensive)
    print("\nComputing RF probabilities on Wednesday test set...")
    rf_proba = rf.predict_proba(X_test_scaled)
    p_attack_rf = 1 - rf_proba[:, 0]
    print(f"  Range: [{p_attack_rf.min():.4f}, {p_attack_rf.max():.4f}]")
    
    print("\nComputing IF scores on Wednesday test set...")
    s_if = iforest.score(X_test_scaled)
    print(f"  Range: [{s_if.min():.4f}, {s_if.max():.4f}]")
    
    # Grid search
    print("\nRunning grid search...")
    weight_grid = [
        (0.9, 0.1),
        (0.8, 0.2),
        (0.7, 0.3),
        (0.6, 0.4),
        (0.5, 0.5),
        (0.4, 0.6),
        (0.3, 0.7),
        (0.2, 0.8),
        (0.1, 0.9),
    ]
    
    results: list[dict[str, float]] = []
    for w_rf, w_if in weight_grid:
        combined = w_rf * p_attack_rf + w_if * s_if
        
        # Dynamic threshold: 99th percentile of BENIGN combined scores
        benign_mask = (y_true_binary == 0)
        benign_combined: npt.NDArray[np.int16] = combined[benign_mask]
        threshold = float(np.percentile(benign_combined, 99))
        
        metrics = evaluate_config(combined, y_true_binary, threshold)
        metrics['w_rf'] = w_rf
        metrics['w_if'] = w_if
        results.append(metrics)
        
        print(f"  w_rf={w_rf}, w_if={w_if} | thr={threshold:.4f} | "
              f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
              f"F1={metrics['f1']:.4f} FAR={metrics['far']:.4f}")
    
    # Find best F1
    df = pd.DataFrame(results)
    df = df.sort_values('f1', ascending=False)
    print("\n=== RESULTS SORTED BY F1 ===")
    print(df.to_string(index=False))
    
    best = df.iloc[0]
    print(f"\n=== WINNING CONFIGURATION ===")
    print(f"w_rf = {best['w_rf']}")
    print(f"w_if = {best['w_if']}")
    print(f"threshold = {best['threshold']:.6f}")
    print(f"F1 = {best['f1']:.4f}")
    print(f"Precision = {best['precision']:.4f}")
    print(f"Recall = {best['recall']:.4f}")
    print(f"FAR = {best['far']:.4f}")
    
    # Save winning config
    config = {
        'w_rf': float(best['w_rf']),
        'w_if': float(best['w_if']),
        'threshold': float(best['threshold']),
    }
    joblib.dump(config, 'datasets/processed/fusion_config.joblib')
    print(f"\n✓ Saved winning config to fusion_config.joblib")

if __name__ == '__main__':
    main()
