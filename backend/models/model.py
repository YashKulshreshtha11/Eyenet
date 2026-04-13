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
import timm
from backend.config import NUM_CLASSES

class EyeNetEnsemble(nn.Module):
    """
    EyeNet Production Ensemble (ResNet50 + EfficientNet-B0 + DenseNet121).
    This architecture is designed to match the provided weights file (fc head, no attention).
    """
    def __init__(self, pretrained: bool = True, num_classes: int = NUM_CLASSES):
        super().__init__()
        
        # 1. Backbones
        r_weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.resnet = resnet50(weights=r_weights)
        self.resnet.fc = nn.Identity()

        # Using timm version to match weights keys: blocks, conv_stem, bn1, etc.
        self.effnet = timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=0)

        d_weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        self.densenet = densenet121(weights=d_weights)
        self.densenet.classifier = nn.Identity()

        # Total combined features: 2048 (ResNet) + 1280 (EffNet) + 1024 (DenseNet) = 4352
        input_dim = 2048 + 1280 + 1024

        # 2. Sequential Classifier (Matching 'fc' in weights)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),      # 0
            nn.BatchNorm1d(512),            # 1
            nn.SiLU(inplace=True),          # 2
            nn.Dropout(0.4),                # 3
            nn.Linear(512, num_classes),    # 4
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract and Pooling
        r_f = self.resnet(x)
        e_f = self.effnet(x)
        d_f = self.densenet(x)

        # Concatenate Features
        combined = torch.cat((r_f, e_f, d_f), dim=1)

        # Classifier head
        return self.fc(combined)

    def freeze_backbones(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.fc.parameters():
            param.requires_grad = True

    def unfreeze_backbones_gradual(self, stage: int = 1):
        if stage == 1:
            for param in self.resnet.layer4.parameters(): param.requires_grad = True
            if hasattr(self.effnet, "blocks"):
                for param in self.effnet.blocks[-3:].parameters(): param.requires_grad = True
                if hasattr(self.effnet, "conv_head"):
                    for param in self.effnet.conv_head.parameters(): param.requires_grad = True
                if hasattr(self.effnet, "bn2"):
                    for param in self.effnet.bn2.parameters(): param.requires_grad = True
            elif hasattr(self.effnet, "features"):
                for param in self.effnet.features[7:].parameters(): param.requires_grad = True
            for param in self.densenet.features.denseblock4.parameters(): param.requires_grad = True
            for param in self.densenet.features.norm5.parameters(): param.requires_grad = True
        else:
            self.unfreeze_backbones()

    def unfreeze_backbones(self):
        for param in self.parameters(): param.requires_grad = True

def build_model(pretrained: bool = True, num_classes: int = NUM_CLASSES) -> EyeNetEnsemble:
    return EyeNetEnsemble(pretrained=pretrained, num_classes=num_classes)
