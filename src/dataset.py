"""PyTorch Dataset over labeled studies, matching the competition's real
on-disk layout: <series_dir>/<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm

Expects train.csv (labels + report text) and train_series.csv (series
metadata: Fluid_Sensitive, Anatomical_Plane) alongside the DICOM folders --
that's exactly what's mounted in the competition's Kaggle Notebook environment
(see infer path documented in notebooks/reference-notes.md).
"""
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from dicom_utils import read_series

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]
PLANE_ORDER = {"Sagittal": 0, "Coronal": 1, "Axial": 2}


def pick_series(series_df, k=3):
    """Choose up to k series for one study: prefer Fluid_Sensitive=1, try to
    get one per anatomical plane before filling remaining slots from the rest.
    Deterministic (no randomness) so train/eval selection is reproducible."""
    df = series_df.copy()
    df["_plane_rank"] = df["Anatomical_Plane"].map(PLANE_ORDER).fillna(3)
    df = df.sort_values(["Fluid_Sensitive", "_plane_rank"], ascending=[False, True])

    picked, seen_planes = [], set()
    for _, row in df[df["Fluid_Sensitive"] == 1].iterrows():
        if row["Anatomical_Plane"] not in seen_planes and len(picked) < k:
            picked.append(row["SeriesInstanceUID"])
            seen_planes.add(row["Anatomical_Plane"])
    for _, row in df.iterrows():
        if len(picked) >= k:
            break
        if row["SeriesInstanceUID"] not in picked:
            picked.append(row["SeriesInstanceUID"])
    return picked


def train_val_split(df, val_frac=0.2, seed=42):
    """Plain random split (not stratified per-label -- with 12 labels and
    only ~58 studies, stratifying on all of them at once isn't really
    possible; a fixed seed at least makes the split reproducible run to run).
    """
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_val = max(1, int(round(len(shuffled) * val_frac)))
    return shuffled.iloc[n_val:].reset_index(drop=True), shuffled.iloc[:n_val].reset_index(drop=True)


class KneeStudyDataset(Dataset):
    def __init__(self, train_csv, train_series_csv, series_root, k_series=3,
                 img_size=128, max_slices=32, only_labeled=True, cache=True):
        train = pd.read_csv(train_csv)
        self.series_df = pd.read_csv(train_series_csv)
        self.series_root = series_root
        self.k_series = k_series
        self.img_size = img_size
        self.max_slices = max_slices

        if only_labeled:
            train = train.dropna(subset=LABELS, how="all")
        self.studies = train.reset_index(drop=True)

        # DICOM decode + resize is the expensive part of __getitem__, and the
        # images never change between epochs -- without this, a 20-epoch run
        # re-decodes every study's DICOMs from scratch 20 times over for no
        # reason. Small enough dataset here (tens of studies, ~160x160) to
        # just hold every study's tensors in memory after the first read.
        self._cache = {} if cache else None

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, idx):
        if self._cache is not None and idx in self._cache:
            return self._cache[idx]

        row = self.studies.iloc[idx]
        study_uid = row["StudyInstanceUID"]
        label_vals = pd.to_numeric(row[LABELS], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        label = torch.from_numpy(label_vals)

        study_series = self.series_df[self.series_df["StudyInstanceUID"] == study_uid]
        series_uids = pick_series(study_series, k=self.k_series)

        tensors = []
        for suid in series_uids:
            series_dir = os.path.join(self.series_root, study_uid, suid)
            vol = read_series(series_dir, img_size=self.img_size, max_slices=self.max_slices)
            if vol is None:
                continue
            tensors.append(torch.from_numpy(vol).unsqueeze(1))  # (S, 1, H, W)

        result = (tensors, label)
        if self._cache is not None:
            self._cache[idx] = result
        return result


def collate_studies(batch):
    """Studies have a variable number of series/slices, so a batch stays a
    plain list of (series_list, label) pairs rather than a stacked tensor --
    KneeMILModel.forward_batch consumes that list directly."""
    series_lists = [b[0] for b in batch]
    labels = torch.stack([b[1] for b in batch])
    return series_lists, labels
