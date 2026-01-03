import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm


from src.uncertainty.mc_dropout import mc_dropout_predict_proba
from src.evaluation.abstention import risk_coverage_curve, coverage_at_accuracy
from src.evaluation.plots import plot_risk_coverage

from src.data.svhn import get_svhn_loader
from src.evaluation.ood import ood_metrics
from src.evaluation.ood_plots import plot_ood_hist, plot_roc_curve


from src.utils.seed import set_seed
from src.utils.config import load_yaml, get
from src.data.cifar10 import get_cifar10_loaders
from src.models.baseline import build_model
from src.evaluation.metrics import accuracy_from_logits, as_dict
from src.evaluation.predict import predict_proba
from src.calibration.ece import compute_ece
from src.calibration.plots import (
    plot_confidence_hist,
    plot_reliability_diagram,
)


def pick_device(device_str: str) -> torch.device:
    if device_str == "cpu":
        return torch.device("cpu")
    if device_str == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    # ---------------- Argument & config ----------------
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    # ---------------- Data ----------------
    data_dir = get(cfg, "data.data_dir", "data")
    batch_size = int(get(cfg, "data.batch_size", 128))
    num_workers = int(get(cfg, "data.num_workers", 2))

    # ---------------- Model ----------------
    model_name = get(cfg, "model.name", "resnet18")
    num_classes = int(get(cfg, "model.num_classes", 10))
    pretrained = bool(get(cfg, "model.pretrained", False))

    # ---------------- Training ----------------
    epochs = int(get(cfg, "train.epochs", 10))
    lr = float(get(cfg, "train.lr", 1e-3))
    wd = float(get(cfg, "train.weight_decay", 5e-4))
    device = pick_device(get(cfg, "train.device", "auto"))
    unc_method = get(cfg, "uncertainty.method", None)
    n_passes = int(get(cfg, "uncertainty.n_passes", 20))
    abst_metric = get(cfg, "uncertainty.abstain.metric", "predictive_entropy")

    # ---------------- Output ----------------
    run_root = get(cfg, "output.run_dir", "experiments/runs")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_baseline")
    run_dir = os.path.join(run_root, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # ---------------- Loaders ----------------
    train_loader, val_loader = get_cifar10_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # ---------------- Model & Optimizer ----------------
    model = build_model(
        model_name,
        num_classes=num_classes,
        pretrained=pretrained,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=wd)

    # ---------------- Training loop ----------------
    history = []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, optimizer=optimizer, device=device
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, optimizer=None, device=device
        )

        row = as_dict(epoch, train_loss, train_acc, val_loss, val_acc)
        history.append(row)

        print(
            f"[Epoch {epoch:02d}/{epochs}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

       # ---------------- Prediction for calibration/uncertainty ----------------
    if unc_method == "mc_dropout":
        probs, labels, extras = mc_dropout_predict_proba(
            model, val_loader, device=device, n_passes=n_passes
        )
        abstain_score = extras.get(abst_metric)
        higher_means_more_confident = (abst_metric == "max_prob")

        # ---------------- OOD evaluation (optional) ----------------
        ood_enabled = bool(get(cfg, "ood.enabled", False))
        if ood_enabled:
            ood_data_dir = get(cfg, "ood.data_dir", "data")
            ood_dataset = get(cfg, "ood.dataset", "svhn")

            if ood_dataset.lower() != "svhn":
                raise ValueError("Only SVHN OOD is implemented right now.")

            ood_loader = get_svhn_loader(
                data_dir=ood_data_dir,
                batch_size=batch_size,
                num_workers=num_workers,
            )

            # Get OOD uncertainty scores (predictive entropy by default)
            ood_probs, _, ood_extras = mc_dropout_predict_proba(
                model, ood_loader, device=device, n_passes=n_passes
            )

            id_unc = extras["predictive_entropy"]
            ood_unc = ood_extras["predictive_entropy"]

            metrics = ood_metrics(id_unc, ood_unc)

            with open(os.path.join(run_dir, "ood_auroc.json"), "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)

            plot_ood_hist(
                id_unc,
                ood_unc,
                os.path.join(run_dir, "ood_score_hist.png"),
                title="OOD Detection via Predictive Entropy (MC Dropout)"
            )

            import numpy as _np
            plot_roc_curve(
                _np.array(metrics["fpr"], dtype=float),
                _np.array(metrics["tpr"], dtype=float),
                os.path.join(run_dir, "ood_roc_curve.png"),
                title=f"OOD ROC Curve (AUROC={metrics['auroc']:.3f})"
            )

            print(f"✅ OOD AUROC (entropy, CIFAR10 vs SVHN): {metrics['auroc']:.4f}")


        # Save uncertainty summary
        summary = {
            "method": "mc_dropout",
            "n_passes": n_passes,
            "mean_max_prob": float(extras["max_prob"].mean()),
            "mean_predictive_entropy": float(extras["predictive_entropy"].mean()),
            "abstain_metric": abst_metric,
        }
        with open(os.path.join(run_dir, "uncertainty_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Abstention evaluation
        curve = risk_coverage_curve(
            probs=probs,
            labels=labels,
            abstain_score=abstain_score,
            higher_means_more_confident=higher_means_more_confident,
            n_points=60,
        )
        plot_risk_coverage(curve, os.path.join(run_dir, "risk_coverage_curve.png"),
                           title=f"Risk-Coverage ({abst_metric})")

        cov95, acc95 = coverage_at_accuracy(
            probs=probs,
            labels=labels,
            abstain_score=abstain_score,
            higher_means_more_confident=higher_means_more_confident,
            target_accuracy=0.95,
        )
        with open(os.path.join(run_dir, "abstention_metrics.json"), "w", encoding="utf-8") as f:
            json.dump({"coverage_at_95_acc": cov95, "achieved_accuracy": acc95}, f, indent=2)

        print(f"✅ Abstention: coverage@95%acc = {cov95:.3f} (achieved acc {acc95:.3f})")

    else:
        probs, labels = predict_proba(model, val_loader, device=device)
        abstain_score = probs.max(axis=1)  # default confidence

    # ---------------- Calibration diagnostics ----------------
    conf = probs.max(axis=1)
    ece, details = compute_ece(probs, labels, n_bins=15)

    with open(os.path.join(run_dir, "ece.json"), "w", encoding="utf-8") as f:
        json.dump({"ece": ece, "details": details}, f, indent=2)

    plot_confidence_hist(
        conf,
        os.path.join(run_dir, "confidence_hist.png"),
        title="Confidence Histogram",
    )

    plot_reliability_diagram(
        details,
        os.path.join(run_dir, "reliability_diagram.png"),
        title="Reliability Diagram",
    )

    print(f"✅ ECE: {ece:.4f}")
    print(f"✅ Saved evaluation artifacts in: {run_dir}")


    # ---------------- Save checkpoint ----------------
    ckpt_path = os.path.join(run_dir, "model.pt")
    torch.save(
        {"model_state_dict": model.state_dict(), "config": cfg},
        ckpt_path,
    )

    print(f"✅ Saved checkpoint: {ckpt_path}")
    print(f"✅ Saved metrics: {os.path.join(run_dir, 'metrics.json')}")


if __name__ == "__main__":
    main()
