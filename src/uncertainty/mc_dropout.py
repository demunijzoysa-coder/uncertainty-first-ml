from typing import Tuple, Dict
import numpy as np
import torch
from torch import nn
from tqdm import tqdm


def enable_dropout(model: nn.Module) -> None:
    """
    Enable dropout layers during evaluation for MC Dropout.
    """
    for m in model.modules():
        if m.__class__.__name__.lower().startswith("dropout"):
            m.train()


@torch.no_grad()
def mc_dropout_predict_proba(
    model: nn.Module,
    loader,
    device: torch.device,
    n_passes: int = 20,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Returns:
      mean_probs: (N, C)
      labels: (N,)
      extras: dict with predictive_entropy (N,), max_prob (N,)
    """
    model.eval()
    enable_dropout(model)  # key trick

    probs_passes = []
    labels_all = None

    for _ in range(n_passes):
        probs_list = []
        labels_list = []

        for x, y in tqdm(loader, desc="mc-predict", leave=False):
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            probs_list.append(probs)
            labels_list.append(y.numpy())

        probs_passes.append(np.concatenate(probs_list, axis=0))

        if labels_all is None:
            labels_all = np.concatenate(labels_list, axis=0)

    probs_passes = np.stack(probs_passes, axis=0)   # (T, N, C)
    mean_probs = probs_passes.mean(axis=0)           # (N, C)

    max_prob = mean_probs.max(axis=1)                # (N,)
    # predictive entropy: -sum p log p
    eps = 1e-12
    predictive_entropy = -np.sum(mean_probs * np.log(mean_probs + eps), axis=1)

    extras = {
        "max_prob": max_prob,
        "predictive_entropy": predictive_entropy,
    }
    return mean_probs, labels_all, extras
