import torch.nn as nn
from torchvision import models


def create_resnet18_model(num_classes=2):
    model = models.resnet18(weights='DEFAULT')
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model