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

The dataset is ~570 GB of DICOMs, and this machine has no GPU. So the split is:

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

## Setup (local)

```
pip install -r requirements.txt
# Kaggle API token saved to ~/.kaggle/access_token (not part of this repo)
kaggle competitions download -c rsna-knee-abnormality-detection -f train.csv -p data/
kaggle competitions download -c rsna-knee-abnormality-detection -f train_series.csv -p data/
python src/eda.py
```

## Status (2026-08-09)

Repo scaffolded, Kaggle API working, metadata pulled locally. First real numbers:

- 4,407 training studies, only **58** carry the 12 expert labels — the rest have
  only the free-text report. This is the central problem to solve (see
  `notebooks/reference-notes.md`).
- 24,371 series total, ~5.5 series per study on average.
- Label prevalence among the 58 labeled studies is fairly balanced (roughly
  15-60% positive per label, Effusion most common, MCL least common) — no
  extreme class imbalance to fight there, at least on this tiny labeled slice.

Next: build the smallest possible end-to-end pipeline (train on the 58 labeled
studies only, one series per study, simple CNN) to get a real `submission.csv`
out the door, before touching label extraction from reports or fancier models.

## License

Code in this repo is MIT-licensed (see `LICENSE`). The competition data itself
is *not* covered by that license and stays subject to the RSNA MIRA license
referenced on the [competition data page](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data) —
this repo never contains DICOMs or other competition data, only code.
