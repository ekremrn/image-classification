import timm
import torch.nn as nn


def get(arch, not_pretrained, num_classes):
    model = timm.create_model(arch, pretrained=not_pretrained, num_classes=num_classes)
    return model
