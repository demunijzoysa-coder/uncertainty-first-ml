from typing import Dict, Tuple
import numpy as np


def risk_coverage_curve(
    probs: np.ndarray,
    labels: np.ndarray,
    abstain_score: np.ndarray,
    higher_means_more_confident: bool,
    n_points: int = 50,
) -> Dict[str, np.ndarray]:
    """
    Build a risk-coverage curve by varying the abstention threshold.
    - Coverage = fraction answered
    - Risk = 1 - accuracy on answered samples
    abstain_score: confidence-like score (higher = more confident) OR uncertainty-like score (higher = more uncertain)
    """
    # Convert score into "confidence score" where higher = more confident
    if higher_means_more_confident:
        conf_score = abstain_score
    else:
        conf_score = -abstain_score

    # Sort by confidence descending
    order = np.argsort(-conf_score)
    probs_s = probs[order]
    labels_s = labels[order]

    preds = probs_s.argmax(axis=1)
    correct = (preds == labels_s).astype(np.float32)

    N = len(labels)
    coverages = []
    risks = []

    # Evaluate for different answer set sizes
    ks = np.linspace(1, N, n_points).astype(int)
    ks = np.unique(ks)

    for k in ks:
        answered = correct[:k]
        acc = float(answered.mean())
        risk = 1.0 - acc
        coverage = k / N
        coverages.append(coverage)
        risks.append(risk)

    return {
        "coverage": np.array(coverages, dtype=float),
        "risk": np.array(risks, dtype=float),
    }


def coverage_at_accuracy(
    probs: np.ndarray,
    labels: np.ndarray,
    abstain_score: np.ndarray,
    higher_means_more_confident: bool,
    target_accuracy: float = 0.95,
) -> Tuple[float, float]:
    """
    Returns (best_coverage, achieved_accuracy) for the largest coverage achieving >= target_accuracy.
    """
    if higher_means_more_confident:
        conf_score = abstain_score
    else:
        conf_score = -abstain_score

    order = np.argsort(-conf_score)
    probs_s = probs[order]
    labels_s = labels[order]
    preds = probs_s.argmax(axis=1)
    correct = (preds == labels_s).astype(np.float32)

    best_cov = 0.0
    best_acc = 0.0
    N = len(labels)

    running_correct = 0.0
    for i in range(N):
        running_correct += correct[i]
        k = i + 1
        acc = running_correct / k
        cov = k / N
        if acc >= target_accuracy and cov >= best_cov:
            best_cov = cov
            best_acc = acc

    return float(best_cov), float(best_acc)
