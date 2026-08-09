# Reference: anatomy of a top public solution (0.903 AUC, rank 18/792, top 2.3%)

Distilled from a public Kaggle write-up shared 2026-08-09 for this competition. Not
our code — a map of the *ideas*, to understand what a strong solution actually does
and to decide, step by step, which pieces are worth building ourselves first. Public
sharing of Competition Code on Kaggle forums/notebooks is explicitly allowed under
the competition rules (see rules section "Public Code Sharing"), so reusing ideas
(not verbatim code) from it is fair game.

## Why this exists: the label problem dominates everything

4,407 training studies, only 58 have expert labels; the rest have only the free-text
report. Whatever raises label quality/coverage on the other 4,349 raises the ceiling
of literally every downstream choice more than any model tweak would. This solution's
core lever is a local open-weights LLM (Qwen3.6-35B via Ollama) prompted with a strict
12-finding JSON schema, iterated against the 58 gold studies until it hit 0.833 mean
accuracy — then fused with a second cross-check model and a third-party public label
dataset, reliability-weighted per class (a source that's bad specifically at
Synovitis gets down-weighted specifically there, not everywhere).

**Takeaway for us:** don't expect to beat this without solving the label-derivation
problem too — but a 35B local LLM isn't required to get *some* signal from reports;
even simple keyword/regex rules per finding, or a smaller hosted model, would beat
"only train on 58 labels."

## Per-study series selection
Prefer series flagged `Fluid_Sensitive=1` (most informative for fluid/cartilage/
marrow findings), try to get one series per anatomical plane (sagittal/coronal/
axial) before filling remaining slots — same function used at train and inference
time, so training/serving never drift apart.

## DICOM → tensor preprocessing
- Apply `RescaleSlope`/`RescaleIntercept`; invert `MONOCHROME1` images.
- Sort slices by `InstanceNumber` (file names are UIDs, not ordered).
- Per-*series* 1st–99th percentile intensity window (not a global fixed window —
  one bright/dark series shouldn't wreck contrast for the whole batch).
- Center-crop depth to 64 slices, pad-to-square, resize to 288×288, quantize to
  uint8 for a compact cache.

## Augmentation rule with a real reason behind it
5 of 12 labels are laterality-specific (medial vs. lateral meniscus/OA, MCL). A
horizontal flip would silently swap left/right anatomy without updating the label —
loss would still go down normally, so it's a bug you'd never notice from the metrics
alone. Rotation/gamma/scale jitter are shared across a whole series and flips are
never used.

## Model: multi-instance learning (MRNet-style)
Shared 2D CNN backbone (single channel) applied per slice independently → pool
slice embeddings into one per-series embedding (mean / max / gated-attention pool)
→ linear head → 12 logits per series → mean across a study's series. Gated attention
is the fancier option and the code's own default, but a hyperparameter search picked
plain max-pooling for this backbone size/data amount — the fancy option didn't
actually win, so it wasn't used.

## Picking hyperparameters cheaply
20 cheap proxy trials (10% of data, 2 epochs) ranked by 0.7×proxy-CV-AUC +
0.3×proxy-gold58-AUC, before committing to a full 5-fold, 8-epoch run (~12.4h).

## Validation without cheating
Each of the 58 gold studies is scored only by the one fold whose training split
never included it ("cross-fitted"), so the gold58 score can't be inflated by
memorization. Their cross-fitted 0.8568 macro-AUC lined up closely with the
0.8544 cross-validation score — two independent, disjoint checks agreeing.

## Surviving the 9h/no-internet inference budget
- Writes a valid `submission.csv` every 25 studies, not just at the end — a crash
  or hard timeout still leaves a scoreable file.
- Projects total runtime every 25 studies; if projected to blow the budget,
  degrades in stages: first fewer eval slices (32→16), then fewer ensemble folds.
- Any study that fails to decode gets a neutral 0.5 rather than crashing the run.

## Where we go from here
Build up in the same order this write-up implies things get more valuable: get a
submission pipeline working end-to-end first (even on just the 58 gold studies),
then improve labels for the unlabeled majority, then improve the image model, then
worry about ensembling/efficiency. See main `README.md` for the current step.
