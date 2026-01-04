import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from src.models.baseline import build_model
from src.uncertainty.ensemble import ensemble_predict_proba, predictive_entropy


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def load_ensemble(checkpoint_dir, device):
    models = []
    for i in range(5):
        ckpt = torch.load(f"{checkpoint_dir}/model_member_{i}.pt", map_location=device)
        model = build_model("resnet18", num_classes=10, pretrained=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()
        models.append(model)
    return models


def preprocess_image(img: Image.Image):
    tfm = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010),
        ),
    ])
    return tfm(img).unsqueeze(0)


def predict(models, x, device):
    probs = []
    with torch.no_grad():
        for m in models:
            logits = m(x.to(device))
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())

    probs = np.stack(probs, axis=0)
    mean_probs = probs.mean(axis=0)[0]

    pred_idx = int(mean_probs.argmax())
    confidence = float(mean_probs.max())
    entropy = float(-np.sum(mean_probs * np.log(mean_probs + 1e-12)))

    return pred_idx, confidence, entropy
