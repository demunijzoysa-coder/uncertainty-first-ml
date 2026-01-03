from typing import Dict
import torch

@torch.no_grad()
def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean().item()

@torch.no_grad()
def nll_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    # Negative log likelihood (cross-entropy)
    return torch.nn.functional.cross_entropy(logits, y).item()

def as_dict(epoch: int, train_loss: float, train_acc: float, val_loss: float, val_acc: float) -> Dict:
    return {
        "epoch": epoch,
        "train_loss": float(train_loss),
        "train_acc": float(train_acc),
        "val_loss": float(val_loss),
        "val_acc": float(val_acc),
    }
