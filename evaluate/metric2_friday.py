import numpy as np
import numpy.typing as npt

def compute_metrics(predicts: npt.NDArray[np.int16],
                    y_true: npt.NDArray[np.int16]) -> dict[str, float | str]:
    tp = np.sum((predicts == 1) & (y_true == 1))
    fp = np.sum((predicts == 1) & (y_true == 0))
    fn = np.sum((predicts == 0) & (y_true == 1))
    tn = np.sum((predicts == 0) & (y_true == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {'precision': precision, 'recall': recall, 'far': far, 'f1': f1,
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)}