import torch
import torch.nn as nn
from torchvision import models

class CIFARResNet18(nn.Module):
    """ResNet-18 adapted to 32x32 CIFAR images."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.model = models.resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
