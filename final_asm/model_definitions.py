import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

# Model 1: FaceNet with Softmax (Supervised Classification)
class FaceNetSoftmax(nn.Module):
    def __init__(self, num_classes, emb_dim=128):
        super().__init__()
        backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        feat_dim = backbone.fc.in_features

        # Fine-tune only higher layers
        for name, param in backbone.named_parameters():
            param.requires_grad = False
            if "layer4" in name or "bn1" in name or "conv1" in name:
                param.requires_grad = True

        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.embed = nn.Linear(feat_dim, emb_dim)
        self.classifier = nn.Linear(emb_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, return_embedding=False):
        f = self.backbone(x)
        e = nn.functional.normalize(self.embed(f))
        if return_embedding:
            return e
        logits = self.classifier(e)
        probs = self.softmax(logits)
        return logits, probs


# Model 2: FaceNet Triplet (Metric Learning)
class FaceNetTriplet(nn.Module):
    def __init__(self, emb_dim=128):
        super().__init__()
        backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        feat_dim = backbone.fc.in_features

        for name, param in backbone.named_parameters():
            param.requires_grad = False
            if "layer4" in name or "bn1" in name or "conv1" in name:
                param.requires_grad = True

        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.embed = nn.Linear(feat_dim, emb_dim)

    def forward(self, x, return_embedding=False):
        f = self.backbone(x)
        e = nn.functional.normalize(self.embed(f))
        # just return embedding no matter what
        return e
