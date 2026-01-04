from typing import Dict, Tuple, List
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

def predictive_entropy(mean_probs: np.ndarray) -> np.ndarray:
    eps = 1e-12
    return -np.sum(mean_probs * np.log(mean_probs + eps), axis=1)

@torch.no_grad()
def predict_proba_single(model: nn.Module, loader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs_list = []
    labels_list = []

    for x, y in tqdm(loader, desc="ens-predict", leave=False):
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        probs_list.append(probs)
        labels_list.append(y.numpy())

    probs_all = np.concatenate(probs_list, axis=0)
    labels_all = np.concatenate(labels_list, axis=0)
    return probs_all, labels_all

def ensemble_predict_proba(
    models: List[nn.Module],
    loader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Average probabilities across ensemble members.
    """
    probs_members = []
    labels_all = None

    for m in models:
        probs, labels = predict_proba_single(m, loader, device=device)
        probs_members.append(probs)
        if labels_all is None:
            labels_all = labels

    probs_members = np.stack(probs_members, axis=0)  # (M, N, C)
    mean_probs = probs_members.mean(axis=0)          # (N, C)

    max_prob = mean_probs.max(axis=1)
    ent = predictive_entropy(mean_probs)

    extras = {
        "max_prob": max_prob,
        "predictive_entropy": ent,
    }
    return mean_probs, labels_all, extras
