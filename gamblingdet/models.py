"""MLP classifier head and the six fusion strategies compared in the paper."""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .config import DROPOUT, HIDDEN_DIM, PROJ_DIM


def mlp(in_dim):
    return nn.Sequential(nn.Linear(in_dim, HIDDEN_DIM), nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(HIDDEN_DIM, 1))


class MLPClassifier(nn.Module):
    """Unimodal head: Linear(d, 128) -> ReLU -> Dropout(0.3) -> Linear(128, 1). Outputs a logit."""

    def __init__(self, embed_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, HIDDEN_DIM)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT)
        self.fc2 = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, x):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))


class ConcatFusion(nn.Module):
    """Early fusion: MLP on [e_img || e_txt]."""

    def __init__(self, img_dim, txt_dim):
        super().__init__()
        self.classifier = mlp(img_dim + txt_dim)

    def forward(self, img, txt):
        return self.classifier(torch.cat([img, txt], dim=1))


class AverageFusion(nn.Module):
    """Late fusion: mean of the two unimodal logits."""

    def __init__(self, img_dim, txt_dim):
        super().__init__()
        self.img_clf, self.txt_clf = mlp(img_dim), mlp(txt_dim)

    def forward(self, img, txt):
        return (self.img_clf(img) + self.txt_clf(txt)) / 2


class WeightedFusion(nn.Module):
    """Late fusion: softmax-normalised learnable weights over the unimodal logits (init 0.5/0.5)."""

    def __init__(self, img_dim, txt_dim):
        super().__init__()
        self.img_clf, self.txt_clf = mlp(img_dim), mlp(txt_dim)
        self.weights = nn.Parameter(torch.ones(2))

    def forward(self, img, txt):
        w = F.softmax(self.weights, dim=0)
        return w[0] * self.img_clf(img) + w[1] * self.txt_clf(txt)


class StackingFusion(nn.Module):
    """Late fusion: a meta-classifier (2 -> 16 -> 1) on the concatenated unimodal logits."""

    def __init__(self, img_dim, txt_dim):
        super().__init__()
        self.img_clf, self.txt_clf = mlp(img_dim), mlp(txt_dim)
        self.meta = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(16, 1))

    def forward(self, img, txt):
        return self.meta(torch.cat([self.img_clf(img), self.txt_clf(txt)], dim=1))


class ProjectionFusion(nn.Module):
    """Intermediate fusion: Linear -> ReLU -> Dropout projection of each modality to d=512, then concat + MLP."""

    def __init__(self, img_dim, txt_dim, proj_dim=PROJ_DIM):
        super().__init__()
        self.img_proj = nn.Sequential(nn.Linear(img_dim, proj_dim), nn.ReLU(), nn.Dropout(DROPOUT))
        self.txt_proj = nn.Sequential(nn.Linear(txt_dim, proj_dim), nn.ReLU(), nn.Dropout(DROPOUT))
        self.classifier = mlp(proj_dim * 2)

    def forward(self, img, txt):
        return self.classifier(torch.cat([self.img_proj(img), self.txt_proj(txt)], dim=1))


class ContrastiveSimilarityFusion(nn.Module):
    """Intermediate fusion (CLIP-style): logit = tau * cos(W_i e_img, W_t e_txt), tau learnable, init 1/0.07."""

    def __init__(self, img_dim, txt_dim, proj_dim=PROJ_DIM):
        super().__init__()
        self.img_proj = nn.Linear(img_dim, proj_dim)
        self.txt_proj = nn.Linear(txt_dim, proj_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, img, txt):
        z_img = F.normalize(self.img_proj(img), dim=-1)
        z_txt = F.normalize(self.txt_proj(txt), dim=-1)
        return (z_img * z_txt).sum(dim=-1, keepdim=True) * self.logit_scale.exp()


FUSION_MODELS = {
    "concat": ConcatFusion,
    "average": AverageFusion,
    "weighted": WeightedFusion,
    "stacking": StackingFusion,
    "projection": ProjectionFusion,
    "contrastive": ContrastiveSimilarityFusion,
}

FUSION_NAMES = {
    "concat": "Concatenation",
    "average": "Average",
    "weighted": "Weighted",
    "stacking": "Stacking",
    "projection": "Projection",
    "contrastive": "Contrastive Similarity",
}
