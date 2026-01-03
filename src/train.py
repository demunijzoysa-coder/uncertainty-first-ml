import argparse
import json
import os
from datetime import datetime

import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from src.utils.seed import set_seed
from src.utils.config import load_yaml, get
from src.data.cifar10 import get_cifar10_loaders
from src.models.baseline import build_model
from src.evaluation.metrics import accuracy_from_logits, as_dict

def pick_device(device_str: str) -> torch.device:
    if device_str == "cpu":
        return torch.device("cpu")
    if device_str == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_epoch(model: nn.Module, loader, optimizer=None, device=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for x, y in tqdm(loader, desc="train" if is_train else "eval", leave=False):
        x, y = x.to(device), y.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_acc += accuracy_from_logits(logits.detach(), y)
        n_batches += 1

    return total_loss / n_batches, total_acc / n_batches

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    data_dir = get(cfg, "data.data_dir", "data")
    batch_size = int(get(cfg, "data.batch_size", 128))
    num_workers = int(get(cfg, "data.num_workers", 2))

    model_name = get(cfg, "model.name", "resnet18")
    num_classes = int(get(cfg, "model.num_classes", 10))
    pretrained = bool(get(cfg, "model.pretrained", False))

    epochs = int(get(cfg, "train.epochs", 10))
    lr = float(get(cfg, "train.lr", 1e-3))
    wd = float(get(cfg, "train.weight_decay", 5e-4))
    device = pick_device(get(cfg, "train.device", "auto"))

    run_root = get(cfg, "output.run_dir", "experiments/runs")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_baseline")
    run_dir = os.path.join(run_root, run_id)
    os.makedirs(run_dir, exist_ok=True)

    train_loader, val_loader = get_cifar10_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers
    )

    model = build_model(model_name, num_classes=num_classes, pretrained=pretrained).to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=wd)

    history = []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, optimizer=optimizer, device=device)
        val_loss, val_acc = run_epoch(model, val_loader, optimizer=None, device=device)

        row = as_dict(epoch, train_loss, train_acc, val_loss, val_acc)
        history.append(row)

        print(f"[Epoch {epoch:02d}/{epochs}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    ckpt_path = os.path.join(run_dir, "model.pt")
    torch.save({"model_state_dict": model.state_dict(), "config": cfg}, ckpt_path)
    print(f"✅ Saved checkpoint: {ckpt_path}")
    print(f"✅ Saved metrics: {os.path.join(run_dir, 'metrics.json')}")

if __name__ == "__main__":
    main()
