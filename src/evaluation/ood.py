from typing import Dict
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score

def ood_metrics(id_scores: np.ndarray, ood_scores: np.ndarray) -> Dict:
    """
    Higher score = more OOD-like (more uncertain).
    id_scores: uncertainty scores for in-distribution samples
    ood_scores: uncertainty scores for OOD samples
    """
    y_true = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
    y_score = np.concatenate([id_scores, ood_scores])

    auroc = roc_auc_score(y_true, y_score)
    aupr = average_precision_score(y_true, y_score)

    fpr, tpr, _ = roc_curve(y_true, y_score)

    return {
        "auroc": float(auroc),
        "aupr": float(aupr),
        "fpr": fpr.astype(float).tolist(),
        "tpr": tpr.astype(float).tolist(),
    }
