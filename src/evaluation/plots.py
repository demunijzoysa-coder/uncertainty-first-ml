import matplotlib.pyplot as plt
import numpy as np

def plot_risk_coverage(curve: dict, out_path: str, title: str = "Risk-Coverage Curve"):
    cov = curve["coverage"]
    risk = curve["risk"]

    plt.figure()
    plt.plot(cov, risk)
    plt.title(title)
    plt.xlabel("Coverage (fraction answered)")
    plt.ylabel("Risk (1 - accuracy on answered)")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
