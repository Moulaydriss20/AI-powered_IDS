import pandas as pd
import numpy as np
import joblib
from src.preprocessing.cleaner import cleaner
from evaluate.metric2_friday import compute_metrics

# ============================================================
# Configuration
# ============================================================
FRIDAY_FILES = {
    'Botnet':   'datasets/raw/Friday-WorkingHours-Morning.pcap_ISCX.csv',
    'PortScan': 'datasets/raw/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv',
    'DDoS':     'datasets/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
}

# From train5.py output — Wednesday-benign 99th-percentile thresholds per weight
WEIGHT_THRESHOLDS = [
    (0.9, 0.1, 0.0558),
    (0.8, 0.2, 0.1113),
    (0.7, 0.3, 0.1664),
    (0.6, 0.4, 0.2216),
    (0.5, 0.5, 0.2768),
    (0.4, 0.6, 0.3321),
    (0.3, 0.7, 0.3873),
    (0.2, 0.8, 0.4425),
    (0.1, 0.9, 0.4965),
]

# ============================================================
# Main
# ============================================================
def main():
    # Load artifacts
    scaler = joblib.load('datasets/processed/scaler.joblib')
    rf = joblib.load('datasets/processed/random_forest_model.joblib')
    iforest = joblib.load('datasets/processed/isolation_forest.joblib')
    
    all_results: list[dict[str, float | str]] = []
    
    for family, csv_path in FRIDAY_FILES.items():
        print(f"\n{'='*60}")
        print(f"Processing {family}: {csv_path}")
        print('='*60)
        
        # Load and preprocess
        df_clean = cleaner(csv_path)
        y_str = df_clean['Label'].values
        y_true = np.array(y_str != 'BENIGN').astype(np.int16)
        X = df_clean.drop(columns=['Label'])
        X_scaled = scaler.transform(X).to_numpy()
        
        n = len(y_true)
        n_attacks = int(np.sum(y_true))
        n_benign = int(n - n_attacks)
        print(f"  Total: {n} | Attacks: {n_attacks} | Benign: {n_benign}")
        
        # Compute RF probabilities and IF scores ONCE
        print(f"  Computing RF probabilities...")
        rf_proba = rf.predict_proba(X_scaled)
        p_attack_rf = 1 - rf_proba[:, 0]
        
        print(f"  Computing IF scores...")
        s_if = iforest.score(X_scaled)
        print(f"  IF score range: [{s_if.min():.4f}, {s_if.max():.4f}]")
        
        # Also compute standalone baselines for comparison
        # RF-alone (predict class then check if != 0)
        rf_class_preds = rf.predict(X_scaled)
        rf_binary = (rf_class_preds != 0).astype(np.int16)
        rf_metrics = compute_metrics(rf_binary, y_true)
        rf_metrics['config'] = 'RF alone'
        rf_metrics['family'] = family
        all_results.append(rf_metrics)
        
        # IF-alone at F1-optimal threshold (0.447)
        if_binary = (s_if >= 0.447).astype(np.int16)
        if_metrics = compute_metrics(if_binary, y_true)
        if_metrics['config'] = 'IF alone (0.447)'
        if_metrics['family'] = family
        all_results.append(if_metrics)
        
        # Fusion for each weight configuration
        for w_rf, w_if, threshold in WEIGHT_THRESHOLDS:
            combined = w_rf * p_attack_rf + w_if * s_if
            predicts = (combined >= threshold).astype(np.int16)
            metrics = compute_metrics(predicts, y_true)
            metrics['config'] = f'Fusion w_rf={w_rf} w_if={w_if}'
            metrics['family'] = family
            metrics['threshold'] = threshold
            all_results.append(metrics)
            print(f"    w_rf={w_rf} w_if={w_if} thr={threshold:.4f} | "
                  f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
                  f"F1={metrics['f1']:.4f} FAR={metrics['far']:.4f}")
    
    # Build results DataFrame
    df = pd.DataFrame(all_results)
    df = df[['family', 'config', 'precision', 'recall', 'far', 'f1', 'tp', 'fp', 'fn', 'tn']]
    
    print(f"\n{'='*60}")
    print("=== COMPLETE RESULTS ===")
    print('='*60)
    for family in FRIDAY_FILES.keys():
        print(f"\n--- {family} ---")
        sub = df[df['family'] == family].copy()
        sub = sub.sort_values('f1', ascending=False)
        print(sub.to_string(index=False))
    
    # Save full table
    df.to_csv('datasets/processed/friday_fusion_results.csv', index=False)
    print(f"\n✓ Saved results to datasets/processed/friday_fusion_results.csv")

if __name__ == '__main__':
    main()