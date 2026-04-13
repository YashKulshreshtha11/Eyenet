from __future__ import annotations

from backend.torch_bootstrap import prepare_torch_environment

prepare_torch_environment()

import torch
import torch.nn as nn
from torchvision.models import (
    DenseNet121_Weights,
    EfficientNet_B0_Weights,
    ResNet50_Weights,
    densenet121,
    efficientnet_b0,
    resnet50,
)

from backend.config import NUM_CLASSES


class EyeNetEnsemble(nn.Module):
    def __init__(self, pretrained: bool = True, num_classes: int = NUM_CLASSES):
        super().__init__()

        resnet_weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        efficientnet_weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        densenet_weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None

        self.resnet = resnet50(weights=resnet_weights)
        self.efficientnet = efficientnet_b0(weights=efficientnet_weights)
        self.densenet = densenet121(weights=densenet_weights)

        self.resnet.fc = nn.Identity()
        self.efficientnet.classifier = nn.Identity()
        self.densenet.classifier = nn.Identity()

        total_features = 2048 + 1280 + 1024
        self.feature_dropout = nn.Dropout(p=0.10)
        self.fusion = nn.Sequential(
            nn.Linear(total_features, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(p=0.45),
            nn.Linear(1024, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(p=0.30),
            nn.Linear(384, num_classes),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        resnet_features = self.resnet(x)
        efficientnet_features = self.efficientnet(x)
        densenet_features = self.densenet(x)
        return torch.cat(
            (resnet_features, efficientnet_features, densenet_features),
            dim=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fused_features = self.feature_dropout(self.forward_features(x))
        return self.fusion(fused_features)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return self.softmax(self.forward(x))

    def freeze_backbones(self) -> None:
        for backbone in (self.resnet, self.efficientnet, self.densenet):
            for parameter in backbone.parameters():
                parameter.requires_grad = False
        for parameter in self.fusion.parameters():
            parameter.requires_grad = True

    def unfreeze_backbones_gradual(self, stage: int = 1) -> None:
        self.freeze_backbones()
        if stage <= 1:
            for parameter in self.resnet.layer4.parameters():
                parameter.requires_grad = True
            for parameter in self.efficientnet.features[-2:].parameters():
                parameter.requires_grad = True
            for parameter in self.densenet.features.denseblock4.parameters():
                parameter.requires_grad = True
            for parameter in self.densenet.features.norm5.parameters():
                parameter.requires_grad = True
        else:
            self.unfreeze_backbones()

    def unfreeze_backbones(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = True


def build_model(pretrained: bool = True, num_classes: int = NUM_CLASSES) -> EyeNetEnsemble:
    return EyeNetEnsemble(pretrained=pretrained, num_classes=num_classes)
