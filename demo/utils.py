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


# Put your GitHub Release direct download URL here:
ENSEMBLE_ZIP_URL = "PASTE_YOUR_RELEASE_ZIP_URL_HERE"


def _download_and_extract(zip_url: str, target_dir: Path):
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir / "ensemble_ckpts.zip"

    if not zip_path.exists():
        urllib.request.urlretrieve(zip_url, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)

    # Optional: keep zip to avoid re-download; or delete it
    # zip_path.unlink(missing_ok=True)


def load_ensemble(checkpoint_dir: str, device):
    checkpoint_dir = Path(checkpoint_dir)

    # If checkpoints are missing, download them
    expected = [checkpoint_dir / f"model_member_{i}.pt" for i in range(5)]
    if not all(p.exists() for p in expected):
        if ENSEMBLE_ZIP_URL == "PASTE_YOUR_RELEASE_ZIP_URL_HERE":
            raise RuntimeError(
                "ENSEMBLE_ZIP_URL is not set. Upload checkpoints to GitHub Releases "
                "and paste the direct download URL into demo/utils.py."
            )
        _download_and_extract(ENSEMBLE_ZIP_URL, checkpoint_dir)

    # Load models
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
