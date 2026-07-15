import numpy as np
import numpy.typing as npt

def evaluate_config(combined_scores: npt.NDArray[np.float32],
                    y_true_binary: npt.NDArray[np.int16],
                    threshold: float) -> dict[str, float]:
    """Compute precision, recall, F1, FAR at a given threshold."""
    predicts = (combined_scores >= threshold).astype(np.int16)
    
    tp = np.sum((predicts == 1) & (y_true_binary == 1))
    fp = np.sum((predicts == 1) & (y_true_binary == 0))
    fn = np.sum((predicts == 0) & (y_true_binary == 1))
    tn = np.sum((predicts == 0) & (y_true_binary == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'far': far,
        'f1': f1,
        'threshold': threshold,
    }