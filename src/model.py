"""Baseline model: multi-instance learning over a study's MRI series.

A study has several series (different planes/sequences), each a stack of
slices. We don't have per-slice or per-series labels, only one label per
study, so this is a classic MIL setup (same family as the MRNet paper):

    slice -> shared 2D CNN -> per-slice embedding
    per-slice embeddings (within one series) -> max-pool -> per-series embedding
    per-series embedding -> linear head -> 12 logits for that series
    per-series logits (within one study) -> mean -> final 12 logits for the study

Max-pooling over slices/series is a deliberate simplification of gated
attention pooling (which the original solution's own default is, per
notebooks/reference-notes.md) -- start simple, only add attention if a
held-out comparison actually shows it helps.
"""
import torch
import torch.nn as nn

NUM_LABELS = 12


class SliceEncoder(nn.Module):
    """One grayscale MRI slice -> a fixed-size embedding vector."""

    def __init__(self, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, embed_dim, 3, padding=1), nn.BatchNorm2d(embed_dim), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        """x: (n_slices, 1, H, W) -> (n_slices, embed_dim)."""
        return self.net(x).flatten(1)


class KneeMILModel(nn.Module):
    def __init__(self, num_labels=NUM_LABELS, embed_dim=128):
        super().__init__()
        self.encoder = SliceEncoder(embed_dim)
        self.head = nn.Linear(embed_dim, num_labels)

    def forward_series(self, slices):
        """slices: (n_slices, 1, H, W) -> (num_labels,) logits for one series."""
        embeds = self.encoder(slices)
        pooled, _ = embeds.max(dim=0)
        return self.head(pooled)

    def forward_study(self, series_list):
        """series_list: list of (n_slices_i, 1, H, W) tensors, one per series
        in the study -> (num_labels,) logits for the study."""
        logits = torch.stack([self.forward_series(s) for s in series_list])
        return logits.mean(dim=0)

    def forward_batch(self, studies):
        """studies: list of B study inputs (each a list of series tensors)
        -> (B, num_labels) logits."""
        return torch.stack([self.forward_study(s) for s in studies])
