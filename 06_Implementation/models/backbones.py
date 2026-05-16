"""
Backbone wrappers for DF (ViT-Base) and SF (ResNet-50 as Mamba proxy).

Mamba backbone omitted as it requires custom CUDA kernels (causal-conv1d, mamba-ssm).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ViTBackbone(nn.Module):
    """ViT-Base backbone for DF task. Pretrained on ImageNet-21k."""

    def __init__(self, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        from transformers import ViTModel
        if pretrained:
            self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        else:
            from transformers import ViTConfig
            config = ViTConfig(patch_size=16, hidden_size=768, num_hidden_layers=12, num_attention_heads=12)
            self.vit = ViTModel(config)
        self.dim = 768
        if freeze_backbone:
            for p in self.vit.parameters():
                p.requires_grad = False

    def forward(self, x):
        out = self.vit(pixel_values=x)
        return out.last_hidden_state[:, 0, :]


class ResNetBackbone(nn.Module):
    """ResNet-50 backbone for SF task. Proxy for Mamba when CUDA kernels unavailable."""

    def __init__(self, pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        import torchvision.models as models
        self.resnet = models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
        self.dim = 2048
        self.resnet.fc = nn.Identity()
        if freeze_backbone:
            for p in self.resnet.parameters():
                p.requires_grad = False

    def forward(self, x):
        return self.resnet(x)


class EDLClassifier(nn.Module):
    """Full model: Backbone + Evidence Head (for CE / EDL / ORCU / EDL+ORCU modes)."""

    def __init__(self, backbone: nn.Module, num_classes: int = 4, dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        from models.evidence_head import EvidenceHead
        self.head = EvidenceHead(backbone.dim, num_classes, dropout)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

    def predict(self, x):
        features = self.backbone(x)
        return self.head.predict(features)


class CumulativeClassifier(nn.Module):
    """Backbone + Cumulative Ordinal Head (Coral-style K-1 binary)."""

    def __init__(self, backbone: nn.Module, num_classes: int = 4, dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        from models.ordinal_head import CumulativeHead
        self.cum_head = CumulativeHead(backbone.dim, num_classes, dropout)
        self.num_classes = num_classes
        self._ordinal_logits = None

    def forward(self, x):
        features = self.backbone(x)
        self._ordinal_logits = self.cum_head(features)          # (B, K-1)
        z = self._ordinal_to_class_logits(self._ordinal_logits)  # (B, K)
        return None, z

    @property
    def ordinal_logits(self):
        return self._ordinal_logits

    def _ordinal_to_class_logits(self, ol):
        """Convert K-1 cumulative logits to K-class logits for eval compatibility."""
        cum = torch.sigmoid(ol)                                 # P(y > k), (B, K-1)
        p = []
        p.append(1.0 - cum[:, 0:1])                             # P(y = 0)
        for k in range(1, self.num_classes - 1):
            p.append(cum[:, k-1:k] - cum[:, k:k+1])             # P(y = k)
        p.append(cum[:, -1:])                                    # P(y = K-1)
        probs = torch.cat(p, dim=1)
        return torch.log(probs + 1e-8)


def build_model(task: str = "df", mode: str = "edl_orcu", pretrained: bool = True):
    """Factory function: build model by task type and training mode."""
    if task == "df":
        backbone = ViTBackbone(pretrained=pretrained)
    else:
        backbone = ResNetBackbone(pretrained=pretrained)

    if mode == "cumulative":
        return CumulativeClassifier(backbone)
    else:
        return EDLClassifier(backbone)
