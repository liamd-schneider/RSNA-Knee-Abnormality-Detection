# RSNA Knee Abnormality Detection

Learning project + submission for the Kaggle competition
[RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection).

Goal: understand how a real multimodal medical-imaging ML pipeline is built (DICOM
handling, weak/report-derived labels, multi-series aggregation, CNN/transformer
backbones, evaluation), not primarily to win prizes.

## Task

Predict 12 per-study probabilities from knee MRI DICOM series + the free-text
radiology report:

`ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA,
Effusion, Synovitis, Baker's, Contusion, Fracture`

Metric: macro-averaged ROC AUC across the 12 labels.

## Why the workflow looks like this

The dataset is ~570 GB of DICOMs, and this machine has no GPU / local Python
setup. So the split is:

- **Kaggle Notebooks** (browser, free GPU quota, data pre-mounted): where actual
  EDA-on-full-data, training, and submission notebooks run.
- **This repo (local, synced to GitHub)**: source of truth for code — notebooks
  get exported/downloaded from Kaggle and committed here, reusable code lives in
  `src/`, and this README/`notes/` track what we tried and learned.

## Layout

- `notebooks/` — exported Kaggle notebooks (EDA, baseline, experiments)
- `src/` — reusable Python modules (dataset loading, model, training loop)
- `data/` — small local artifacts only (e.g. `train.csv`, `train_series.csv`
  metadata pulled via the Kaggle API); actual DICOMs are NOT stored here
  (gitignored, too large — work with them inside Kaggle Notebooks)

## Status

Just getting started (2026-08-09). See commit history / notes for progress.
