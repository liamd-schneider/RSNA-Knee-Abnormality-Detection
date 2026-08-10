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

- `notebooks/` — exported Kaggle notebooks (EDA, baseline, experiments) and
  `reference-notes.md`, a distilled write-up of a public top-scoring solution
  used as a rough roadmap
- `src/` — reusable Python modules (DICOM reading, dataset, model, EDA,
  log-plotting) — the "clean" version of the code, used for local dev/testing
- `kernels/<name>/` — self-contained scripts actually pushed to Kaggle to run
  with GPU + the full competition data mounted (`kaggle kernels push`); each
  is functionally the same code as `src/`, just inlined into one file
- `results/<name>/` — committed outputs from a kernel run (loss curve plots,
  example predictions) so progress is visible without re-running anything
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

## Results

### v1 — baseline MIL model, 58 labeled studies only

A from-scratch CNN (no pretrained weights), multi-instance-learning over up to
3 series/study, trained for 20 epochs on just the 58 expert-labeled studies.
Goal for v1 was purely "does the whole pipeline run end to end and produce a
valid submission" — not a competitive score.

![training loss curve](results/v1/loss_curve.png)

Training loss falls steadily (0.638 → 0.593) and hadn't plateaued yet at
epoch 20 — the run was mechanically healthy. But a closer look at the
predictions (`results/v1/submission_example.csv`) shows what that loss number
doesn't: the model's *average* predicted probability per label correlates
**0.95** with that label's raw prevalence in the 58 training studies. In other
words, at this stage it has mostly learned "guess close to the training
average for each label," not yet real per-image visual signal — expected and
diagnosable with only 58 training examples, a from-scratch backbone, and no
held-out validation yet to even measure real generalization. That gap is
exactly what the next steps (validation split, more training data via
report-derived labels, a pretrained backbone) are meant to close.

### v2 — held-out validation

Same model, but an honest 80/20 train/val split (46/12 studies) instead of
training loss as the only signal, with per-label ROC AUC computed on the
held-out 12 (labels with only one class present in that tiny split are
skipped rather than silently scored as a meaningless 0.5/undefined AUC).

### v3 — caching + a real GPU

v2 ran at ~90s/epoch. Two fixes, both driven by profiling that number rather
than guessing:

- **In-memory tensor caching.** v1/v2 re-decoded every study's DICOMs from
  scratch on *every* epoch, even though the images never change between
  epochs. Measured locally: ~2.4s to decode one study cold, <0.001ms to read
  it back from an in-memory cache after that — decoding was almost certainly
  the actual bottleneck, not the (tiny) model itself.
- **A GPU that's actually used.** `enable_gpu: true` got us a P100, but its
  compute capability (sm_60) isn't supported by the CUDA kernels in Kaggle's
  pre-installed PyTorch build, so v1/v2 silently fell back to CPU *while
  still burning P100 quota hours for nothing*. v3 installs an older
  torch+cu121 build (`torch==2.3.1`, still ships sm_60 kernels) before torch
  is ever imported. Confirmed working: `torch_version: "2.3.1+cu121"`,
  `device: "cuda"` in `results/v3/history_v3.json`. (Needs internet enabled
  for this dev/training kernel specifically — never for the eventual
  no-internet submission notebook.)

Concrete result: v3 finished its full 20-epoch run *before v2 did*, on the
same P100 class of hardware v2 had silently downgraded away from.

![train loss vs val AUC](results/v3/history_curve.png)

This plot is the real payoff of adding validation in v2: it's a textbook
overfitting curve. Training loss falls smoothly the entire time (0.652 →
0.572), while held-out validation AUC never crosses the 0.5 random-guess
line and shows no upward trend, drifting noisily between roughly 0.34 and
0.43. The model is getting better at fitting the 46 training studies and
*not* better at generalizing to studies it hasn't seen — exactly what you'd
expect from a from-scratch CNN with no pretrained features and only 46
training examples for a 12-label problem. Per-label AUC on the final epoch
swings wildly (`PF OA` 0.69, `Medial OA` 0.09) which is itself informative:
with only ~12 validation studies, per-label AUC is estimated from a handful
of examples and is far too noisy to read individual numbers as meaningful —
only the aggregate pattern (consistently at/below 0.5, no trend) is.

This is the concrete, measured case for the next two planned steps: more
training data (report-derived labels for the other 4,349 studies) and a
pretrained backbone, in that order — a from-scratch CNN on 46 examples
structurally cannot generalize much better than this, no matter how long
it trains.

### Report-derived labels (v1) — validated before trusting

4,349 of the 4,407 studies have no expert label, only a free-text report
(Spanish, French, English, and others — `kernels/extract_labels_v1/extract.py`
found that `train.csv` is **latin-1 encoded, not UTF-8**; reading it with
pandas' default silently mangled every accented character in that column
instead of raising). `notebooks/reference-notes.md`'s top-scoring public
solution used a large LLM to turn those reports into labels; the same idea
here, sized to what actually runs on Kaggle's free GPU: `Qwen/Qwen2.5-3B-Instruct`,
batched, greedy-decoded, with a hand-written prompt covering all 12
findings' clinical definitions (broadened Synovitis to include Hoffa fat
pad impingement/plica syndrome, effusion counts as present even at "trace"
amounts, Contusion requires a described traumatic pattern rather than
ordinary degenerative marrow edema, Fracture includes avulsion/insufficiency
fractures).

Same rule as the reference solution: never trust an extraction prompt on
the unlabeled studies before checking it against the studies that *do* have
real labels. `kernels/extract_labels_v1` runs the extractor on only the 58
gold-labeled studies and scores it against their real labels first:

- **75.9% mean accuracy, 0.64 mean F1** across the 12 labels
  (`results/extract_labels_v1/validation_report.json`) — smaller than the
  reference solution's 35B-model result (83.3%), which tracks with using a
  ~10x smaller, free-to-run model.
- 11 of 12 labels clearly beat the naive "always predict the majority
  class" baseline — real signal, not noise. The one exception is MCL
  (81.0% vs. an 84.5% majority baseline: MCL is rare enough in this sample
  that always guessing "no MCL injury" edges out the model on raw accuracy,
  though its 0.52 F1 shows it is finding real positives, just with a
  costly false-positive rate).
- ~10% of the 58 reports didn't come back as valid JSON and were
  conservatively defaulted to all-negative rather than silently guessed.

Good enough to trust as a *weaker, soft* label source for the 4,349
unlabeled studies — not as good as the 58 real expert labels, but a real
step up from training on 46 studies alone. `kernels/extract_labels_v1_full`
runs the same, already-validated prompt over those 4,349 studies.

That full run finished after ~4.7h at a steady ~0.24 studies/s (batched
generation, 3B model on a P100): **4,349/4,349 studies processed, 369
(8.5%) parse failures** — consistent with the 58-study validation run's
~10%, so the failure rate isn't an artifact of the smaller sample. 3,980
studies came back with usable extracted labels, feeding directly into
`kernels/train_v4` capped at 1,500 of them (see above for why not all
3,980). One real gap found running this at full scale: the script only
writes its output CSV once, after all 4,349 studies are done, rather than
checkpointing incrementally — a run killed by Kaggle's session time limit
partway through would have lost everything rather than a partial result.
Worth fixing before running anything this long again.

### v4 — same model, 34x more training studies

46 gold + 1,500 extracted-label studies (1,546 total) vs. v3's 46 alone,
same from-scratch CNN, validated on the same 12 held-out gold studies.
Cost some real debugging first: the first attempt failed outright
(`enable_internet` was accidentally left `false`, blocking the same
torch/P100 install v3 needs); the second died silently ~38 minutes in with
no traceback, and the third died again at a different point with the same
no-traceback pattern — both consistent with an out-of-space/OOM kill from
the disk-backed tensor cache (a guessed, unconfirmed `/kaggle/temp/...`
path, uncompressed `.npz` files, no ceiling on writes). Fixed by moving the
cache to `tempfile.gettempdir()`, switching to `np.savez_compressed`,
disabling further cache writes once free disk drops below 3 GB instead of
crashing, and wrapping each study's training step in try/except so one bad
study can't take an hours-long run down with it. Also found and fixed,
separately: a single DICOM in the larger corpus had corrupted pixel data
and crashed `read_series` outright — v1-v3 never hit this because they only
ever touched the 58 gold studies' DICOMs.

![train loss vs val AUC](results/v4/history_curve.png)

With that fixed, the run itself: **best val macro AUC 0.541 at epoch 4**
(vs. v3's best of 0.430, and v3 never once crossed the 0.5 random line).
Real, if modest, improvement — but not a clean one. The curve peaks at
epoch 4 and drifts back down to 0.457 by epoch 8, and per-label AUC at the
final epoch is all over the place (`Medial OA` 0.66, `Effusion` 0.19).
Given only 12 validation studies, individual epoch-to-epoch and per-label
swings are still mostly noise (see v3's writeup) — the one thing worth
reading here is the aggregate level: v4 hovers *around* 0.5 instead of
*below* it like v3 consistently did, which is a real if modest signal that
more data helps, but 34x more data still isn't enough to fix a
from-scratch CNN's fundamental problem. That's exactly the case for
DINOv2 (v5, next): a pretrained backbone is a different lever than more
data, not a substitute for it, and the plan is both, not one instead of
the other.

## License

Code in this repo is MIT-licensed (see `LICENSE`). The competition data itself
is *not* covered by that license and stays subject to the RSNA MIRA license
referenced on the [competition data page](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data) —
this repo never contains DICOMs or other competition data, only code.
