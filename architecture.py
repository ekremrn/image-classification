import timm
import torch.nn as nn


def get(arch, pretrained, num_classes):
    model = timm.create_model(arch, pretrained = pretrained)
    model.fc = nn.Linear(2048, num_classes)
    return model

