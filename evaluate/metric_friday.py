import numpy as np
import numpy.typing as npt
from src.models.random_forest import RandomForest
from src.models.isolation_forest import IsolationForest
from src.models.hybrid_IDS import Hybrid_IDS

def evaluate_family(
        family_name: str, X_scaled: npt.NDArray[np.float32],
        y_str: npt.NDArray[np.int16],
        rf: RandomForest,
        iforest: IsolationForest,
        hybrid: Hybrid_IDS
):
    y_true = (y_str != 'BENIGN').astype(int)

    #Get predictions.
    rf_preds = rf.predict(X_scaled)
    rf_bin = (rf_preds != 0).astype(int)

    if_scores = iforest.score(X_scaled)
    if_bin = (if_scores >= 0.447).astype(int)

    hybrid_preds = hybrid.predict(X_scaled)
    hybrid_bin = (hybrid_preds != 0).astype(int)
    
    #Metrics helper
    def metrics(pred_bin: npt.NDArray[np.int16], name: str):
        TP = np.sum((pred_bin == 1) & (y_true == 1))
        FN = np.sum((pred_bin == 0) & (y_true == 1))
        FP = np.sum((pred_bin == 1) & (y_true == 0))
        TN = np.sum((pred_bin == 0) & (y_true == 0))

        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        far = FP / (FP + TN) if (FP + TN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0

        print(f"    {name:20s} | recall: {recall:.4f} | False alarams rate: {far:.4f} | Precision: {precision:.4f}")

    print(f"\n=== {family_name} ===")
    metrics(rf_bin, 'RF alone')
    metrics(if_bin, f'IF alone ({iforest.score_threshold})')
    metrics(hybrid_bin, f'Hybrid ({hybrid.anomaly_threshold})')

    #Cascade diagnostic
    rf_missed_attacks = (rf_preds == 0) & (y_true == 1)
    n_missed = np.sum(rf_missed_attacks)
    if n_missed > 0:
        cascade_caught = rf_missed_attacks & (if_scores >= 0.447)
        n_caught = np.sum(cascade_caught)
        catch_rate = n_caught / n_missed
        print(f" Cascade diagnostic: RF missed {n_missed} attacks, IF-at-{hybrid.anomaly_threshold} caught: {n_caught}\n catch rate: {catch_rate:.2%}")
    else:
        print(f" Cascade diagnostic: RF missed 0 attacks (nothing to check)")

