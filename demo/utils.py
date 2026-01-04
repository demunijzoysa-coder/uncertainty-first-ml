import os
from pathlib import Path
import zipfile
import urllib.request

import torch
from src.models.baseline import build_model


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# 🔴 IMPORTANT:
# Replace this with the DIRECT download URL from GitHub Releases
# Example:
# https://github.com/<username>/<repo>/releases/download/demo-ckpts-v1/ensemble_ckpts.zip
ENSEMBLE_ZIP_URL = " https://github.com/demunijzoysa-coder/uncertainty-first-ml/commits/Demo"


def _download_and_extract(zip_url: str, target_dir: Path):
    """
    Download ensemble checkpoints zip and extract into target_dir.
    This runs only once on Streamlit Cloud.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir / "ensemble_ckpts.zip"

    if not zip_path.exists():
        try:
            urllib.request.urlretrieve(zip_url, zip_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to download checkpoints from {zip_url}. "
                f"Make sure the GitHub Release URL is correct and public.\n\n{e}"
            )

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target_dir)
    except zipfile.BadZipFile:
        raise RuntimeError(
            "Downloaded checkpoints zip is corrupted. "
            "Re-upload the zip to GitHub Releases and try again."
        )


def load_ensemble(checkpoint_dir: str, device):
    """
    Load deep ensemble models.
    If checkpoints are missing, download them from GitHub Releases.
    """
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
                "4) Commit + push, then reboot the Streamlit app"
            )
        _download_and_extract(ENSEMBLE_ZIP_URL, checkpoint_dir)

    models = []
    for i in range(5):
        ckpt_path = checkpoint_dir / f"model_member_{i}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=device)

        model = build_model("resnet18", num_classes=10, pretrained=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()
        models.append(model)

    return models
