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


class KneeStudyDataset(Dataset):
    def __init__(self, train_csv, train_series_csv, series_root, k_series=3,
                 img_size=128, max_slices=32, only_labeled=True):
        train = pd.read_csv(train_csv)
        self.series_df = pd.read_csv(train_series_csv)
        self.series_root = series_root
        self.k_series = k_series
        self.img_size = img_size
        self.max_slices = max_slices

        if only_labeled:
            train = train.dropna(subset=LABELS, how="all")
        self.studies = train.reset_index(drop=True)

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, idx):
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
        return tensors, label


def collate_studies(batch):
    """Studies have a variable number of series/slices, so a batch stays a
    plain list of (series_list, label) pairs rather than a stacked tensor --
    KneeMILModel.forward_batch consumes that list directly."""
    series_lists = [b[0] for b in batch]
    labels = torch.stack([b[1] for b in batch])
    return series_lists, labels
