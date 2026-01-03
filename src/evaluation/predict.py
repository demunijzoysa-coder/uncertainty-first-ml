from typing import Tuple
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

@torch.no_grad()
def predict_proba(model: nn.Module, loader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs_list = []
    labels_list = []

    for x, y in tqdm(loader, desc="predict", leave=False):
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        probs_list.append(probs)
        labels_list.append(y.numpy())

    probs_all = np.concatenate(probs_list, axis=0)
    labels_all = np.concatenate(labels_list, axis=0)
    return probs_all, labels_all
