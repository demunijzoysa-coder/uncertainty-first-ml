import os
from pathlib import Path
import zipfile
import urllib.request

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from src.models.baseline import build_model


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# 🔴 IMPORTANT:
# Replace this with the DIRECT download URL from GitHub Releases
# Example:
# https://github.com/<username>/<repo>/releases/download/demo-ckpts-v1/ensemble_ckpts.zip
ENSEMBLE_ZIP_URL = "https://github.com/demunijzoysa-coder/uncertainty-first-ml/commits/Demo"


# -----------------------------
# Download + load ensemble
# -----------------------------
def _download_and_extract(zip_url: str, target_dir: Path):
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir / "ensemble_ckpts.zip"

    if not zip_path.exists():
        try:
            urllib.request.urlretrieve(zip_url, zip_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to download checkpoints from {zip_url}. "
                f"Ensure the GitHub Release URL is correct and public.\n\n{e}"
            )

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target_dir)
    except zipfile.BadZipFile:
        raise RuntimeError(
            "Downloaded checkpoints zip is corrupted. "
            "Re-upload the zip to GitHub Releases."
        )


def load_ensemble(checkpoint_dir: str, device):
    checkpoint_dir = Path(checkpoint_dir)

    expected = [checkpoint_dir / f"model_member_{i}.pt" for i in range(5)]
    if not all(p.exists() for p in expected):
        if ENSEMBLE_ZIP_URL == "PASTE_YOUR_RELEASE_ZIP_URL_HERE":
            raise RuntimeError(
                "ENSEMBLE_ZIP_URL is not set.\n\n"
                "Fix:\n"
                "1) Upload ensemble_ckpts.zip to GitHub Releases\n"
                "2) Copy the DIRECT download URL\n"
                "3) Paste it into demo/utils.py as ENSEMBLE_ZIP_URL\n"
                "4) Commit + push, then reboot Streamlit app"
            )
        _download_and_extract(ENSEMBLE_ZIP_URL, checkpoint_dir)

    models = []
    for i in range(5):
        ckpt_path = checkpoint_dir / f"model_member_{i}.pt"
        ckpt = torch.load(ckpt_path, map_location=device)

        model = build_model("resnet18", num_classes=10, pretrained=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()
        models.append(model)

    return models


# -----------------------------
# Image preprocessing
# -----------------------------
def preprocess_image(img: Image.Image) -> torch.Tensor:
    tfm = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010),
        ),
    ])
    return tfm(img).unsqueeze(0)


# -----------------------------
# Prediction + uncertainty
# -----------------------------
def predict(models, x: torch.Tensor, device):
    probs = []

    with torch.no_grad():
        for m in models:
            logits = m(x.to(device))
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())

    probs = np.stack(probs, axis=0)      # (M, 1, C)
    mean_probs = probs.mean(axis=0)[0]   # (C,)

    pred_idx = int(mean_probs.argmax())
    confidence = float(mean_probs.max())
    entropy = float(-np.sum(mean_probs * np.log(mean_probs + 1e-12)))

    return pred_idx, confidence, entropy
