"""
Model + dataset definitions (PyTorch).

Architecture — a shared backbone with a projection to an embedding, then one
linear head per task:

    image -> backbone -> pooled features -> Linear -> embedding (256-d, L2-norm)
                                                        |
                                    +-------------------+-------------------+
                                    |          |             |              |
                              articleType  baseColour     gender     masterCategory

Why multi-head rather than four models:
  * One backbone forward pass serves every attribute -> 4x cheaper at inference.
  * The auxiliary tasks regularise the trunk; colour and category are strongly
    correlated with article type, so they act as useful inductive bias.
  * The embedding is the real prize: it powers visual similarity search, which
    is what "find products that look like this" actually needs.

The embedding is L2-normalised **inside the model**, so cosine similarity is a
plain dot product downstream and the exported ONNX needs no extra ops.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from app.vision_preprocess import CROP_SIZE, IMAGENET_MEAN, IMAGENET_STD, RESIZE_SHORT
from ml import config


# ─── Dataset ──────────────────────────────────────────────────────────
def build_transforms(train: bool):
    """Training augments; eval MUST mirror app/vision_preprocess.py exactly.

    Note the eval transform has no Normalize: normalisation is baked into the
    exported ONNX graph. During training we apply it explicitly (see
    `NormalizedModel`) so train and export see identical inputs.
    """
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(CROP_SIZE, scale=(0.7, 1.0), ratio=(0.8, 1.25)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            transforms.ToTensor(),   # -> [0,1] CHW
        ])
    return transforms.Compose([
        transforms.Resize(RESIZE_SHORT),      # shorter side, bilinear
        transforms.CenterCrop(CROP_SIZE),
        transforms.ToTensor(),
    ])


class FashionDataset(Dataset):
    def __init__(self, manifest_df, train: bool):
        self.df = manifest_df.reset_index(drop=True)
        self.tf = build_transforms(train)
        self.tasks = config.TASKS

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        try:
            img = Image.open(row["image_path"]).convert("RGB")
        except Exception:
            # A corrupt JPEG mid-epoch shouldn't kill a 4-hour training run.
            img = Image.new("RGB", (CROP_SIZE, CROP_SIZE), (128, 128, 128))
        x = self.tf(img)
        y = {t: torch.tensor(int(row[t]), dtype=torch.long) for t in self.tasks}
        return x, y


# ─── Model ────────────────────────────────────────────────────────────
def _build_backbone(name: str):
    from torchvision import models

    if name == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        feat_dim = m.classifier[0].in_features   # 576
        m.classifier = nn.Identity()
        return m, feat_dim
    if name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        feat_dim = m.classifier[1].in_features   # 1280
        m.classifier = nn.Identity()
        return m, feat_dim
    raise ValueError(f"Unknown backbone: {name}")


class FashionNet(nn.Module):
    """Multi-head classifier. Normalisation happens INSIDE forward().

    This placement is load-bearing. An earlier version normalised only in the
    ONNX export wrapper, so the network trained on raw [0,1] tensors and then
    served on (x-mean)/std inputs — a ~4.4x scale shift the backbone had never
    seen. Offline metrics looked fine (they were computed in torch, un-normalised)
    while the deployed model collapsed to near-chance. Keeping the Sub/Div here
    means training, evaluation, and the exported graph all normalise exactly
    once, in the same place, and cannot drift apart.

    `normalize_input=False` exists only to load checkpoints produced by that
    older, broken code path.
    """

    def __init__(self, num_classes: dict[str, int], backbone: str = config.BACKBONE,
                 embedding_dim: int = config.EMBEDDING_DIM, normalize_input: bool = True):
        super().__init__()
        self.backbone, feat_dim = _build_backbone(backbone)
        self.embed = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(feat_dim, embedding_dim),
        )
        self.heads = nn.ModuleDict(
            {task: nn.Linear(embedding_dim, n) for task, n in num_classes.items()}
        )
        self.embedding_dim = embedding_dim
        self.normalize_input = normalize_input
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        """x: NCHW float32 in [0, 1]. Normalisation is applied here."""
        if self.normalize_input:
            x = (x - self.mean) / self.std
        feats = self.backbone(x)
        emb = self.embed(feats)
        emb = F.normalize(emb, p=2, dim=1)          # unit sphere -> cosine == dot
        logits = {task: head(emb) for task, head in self.heads.items()}
        return emb, logits


class ExportWrapper(nn.Module):
    """Flattens FashionNet's (emb, dict) output into a fixed tuple for ONNX.

    It does NOT normalise — FashionNet already did. This class used to be called
    `NormalizedModel` and applied a second normalisation, which is precisely how
    train/serve skew was introduced.
    """

    def __init__(self, net: FashionNet):
        super().__init__()
        self.net = net

    def forward(self, x):
        emb, logits = self.net(x)
        return (emb, *[logits[t] for t in config.TASKS])


# Backwards-compat alias; new code should use ExportWrapper.
NormalizedModel = ExportWrapper


def class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """Inverse-sqrt frequency weighting: tames the long tail without letting a
    50-sample class dominate a 5,000-sample one."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    w = 1.0 / np.sqrt(counts)
    w = w / w.sum() * num_classes         # mean weight ~= 1
    return torch.tensor(w, dtype=torch.float32)
