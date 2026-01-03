from typing import Dict, Any, Tuple
import numpy as np

def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15
) -> Tuple[float, Dict[str, Any]]:
    """
    Expected Calibration Error (ECE) using equal-width bins over confidence.
    probs: (N, C) predicted probabilities
    labels: (N,) true class indices
    """
    conf = probs.max(axis=1)                 # (N,)
    preds = probs.argmax(axis=1)             # (N,)
    acc = (preds == labels).astype(np.float32)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    bin_data = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # include right edge only for last bin
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)

        if mask.sum() == 0:
            bin_data.append({
                "bin": i,
                "lo": float(lo),
                "hi": float(hi),
                "count": 0,
                "avg_conf": None,
                "avg_acc": None
            })
            continue

        avg_conf = float(conf[mask].mean())
        avg_acc = float(acc[mask].mean())
        frac = float(mask.mean())  # count / N
        ece += abs(avg_acc - avg_conf) * frac

        bin_data.append({
            "bin": i,
            "lo": float(lo),
            "hi": float(hi),
            "count": int(mask.sum()),
            "avg_conf": avg_conf,
            "avg_acc": avg_acc
        })

    details = {
        "n_bins": int(n_bins),
        "bin_edges": bin_edges.tolist(),
        "bins": bin_data
    }
    return float(ece), details
