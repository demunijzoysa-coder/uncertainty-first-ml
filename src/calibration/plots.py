from typing import Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt

def plot_confidence_hist(conf: np.ndarray, out_path: str, title: str = "Confidence Histogram"):
    plt.figure()
    plt.hist(conf, bins=20)
    plt.title(title)
    plt.xlabel("Max softmax probability")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_reliability_diagram(ece_details: Dict[str, Any], out_path: str, title: str = "Reliability Diagram"):
    bins = ece_details["bins"]
    xs = []
    ys = []
    counts = []

    for b in bins:
        if b["count"] == 0:
            continue
        xs.append(b["avg_conf"])
        ys.append(b["avg_acc"])
        counts.append(b["count"])

    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)

    plt.figure()
    # Perfect calibration line
    plt.plot([0, 1], [0, 1])
    # Empirical points
    plt.scatter(xs, ys, s=30)
    plt.title(title)
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
