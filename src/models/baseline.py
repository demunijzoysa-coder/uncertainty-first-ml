import torch.nn as nn
from torchvision import models

def build_model(name: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    name = name.lower()

    if name == "resnet18":
        m = models.resnet18(weights=None if not pretrained else models.ResNet18_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    raise ValueError(f"Unknown model name: {name}")
