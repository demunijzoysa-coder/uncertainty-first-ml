import numpy as np
import matplotlib.pyplot as plt

def plot_ood_hist(id_scores: np.ndarray, ood_scores: np.ndarray, out_path: str, title: str):
    plt.figure()
    plt.hist(id_scores, bins=30, alpha=0.7, label="In-distribution (CIFAR-10)")
    plt.hist(ood_scores, bins=30, alpha=0.7, label="OOD (SVHN)")
    plt.title(title)
    plt.xlabel("Uncertainty score")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, out_path: str, title: str):
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1])
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
