"""RSNA Knee Abnormality Detection -- baseline training kernel (v9).

v8 (loss-weighted: gold=1.0, extracted=0.3, v5's 1,500 studies/6 epochs)
scored 0.681, below v5's unweighted 0.700 -- but its curve started slower
(0.486 vs. v5's 0.553 at epoch 1, down-weighting most of the training data
shrinks the gradient signal from it) and briefly *led* v5 at epoch 4-5
before v5 pulled ahead while still climbing. That shape argues "6 epochs
wasn't enough time for the weighted recipe," not "the idea doesn't work" --
v7-moreepochs already showed this same 1,500-study data size tolerates 15
epochs fine (peaked at epoch 11, essentially matching v5).

v9 is the direct, one-variable check: v8's exact loss weights (gold=1.0,
extracted=0.3) and data (1,500 studies), but EPOCHS raised from 6 to 15 to
give the weighted recipe the time v7-moreepochs showed this data size can
use. If v9 surpasses v5's 0.700, the weighting idea works and just needed
more time; if it plateaus below 0.700 even with time to spare, the
weighting itself (or its specific 0.3 ratio) is the limiting factor, not
the epoch budget.

Carries forward v7's best-checkpoint saving (best-val-AUC epoch's weights,
not whichever epoch the run ends on) and v6's per-epoch history
checkpointing + time budget safety net.

Same model/data pipeline otherwise as v5/v6/v7/v8, so any score difference
is attributable to the epoch budget specifically, not a silent
architecture or preprocessing change:

1. A pretrained backbone (DINOv2-small, via Hugging Face transformers)
   instead of v1-v4's from-scratch CNN. Only the last UNFREEZE_LAST
   transformer blocks + the final layernorm are trainable, at a much lower
   learning rate than the head -- the early layers of a self-supervised ViT
   are generic edge/texture filters worth keeping, and there's nowhere near
   enough data here to safely retrain the whole thing. This is the single
   biggest lever identified from the two ~0.9-scoring public notebooks
   analysed for this project (see README) -- v1-v4 all showed the same
   overfitting pattern (loss falls, held-out AUC doesn't) that a randomly
   initialized encoder produces when data is this scarce.

2. True geometric slice ordering. DICOM file names are opaque UIDs and
   InstanceNumber isn't guaranteed to track physical position either --
   v1-v4 sorted by InstanceNumber anyway, which the reference notebooks
   measured as barely-better-than-uncorrelated with anatomy on this exact
   corpus. Every slice's ImagePositionPatient is now projected onto the
   series' slice normal (cross product of the two ImageOrientationPatient
   axes) and sorted on that, falling back to InstanceNumber only when the
   geometry tags are missing.

3. Laterality normalization. Knee anatomy is asymmetric (which meniscus is
   medial vs. lateral flips with the knee's side), and this corpus has no
   fixed left/right convention across studies. Every study is mirrored onto
   a left-knee convention: a horizontal flip for coronal/axial series, a
   slice-order reversal for sagittal series (sagittal stacks are not mirror
   images of each other). Laterality comes from the DICOM Laterality/
   ImageLaterality tag when present, otherwise from the median x of
   ImagePositionPatient across the study's series (patient coordinates: +x
   is the patient's left), left unresolved (no mirroring) inside a 20mm
   dead zone near the midline.

Same robustness fixes as v4 (corrupted-DICOM skip, disk-space-aware cache,
per-study try/except) since this hits the same real corpus.

Local dry run:
    KNEE_SKIP_TORCH_INSTALL=1 KNEE_INPUT_DIR=data/dryrun \
    KNEE_EXTRACTED_CSV=data/dryrun/report_labels_extracted.csv \
    KNEE_OUT_DIR=/tmp/out KNEE_EPOCHS=1 python train.py
"""
import glob
import os
import shutil
import subprocess
import sys
import tempfile

# ---------------------------------------------------------- torch bootstrap
if not os.environ.get("KNEE_SKIP_TORCH_INSTALL"):
    # torch+torchvision+torchaudio installed together, matched versions --
    # transformers imports torchvision internally even for a pure-vision
    # non-CNN model, and a torch-only reinstall leaves a mismatched
    # torchvision behind (found the hard way building kernels/extract_labels_v1).
    CANDIDATES = [
        ("torch==2.3.1", "torchvision==0.18.1", "torchaudio==2.3.1"),
        ("torch==2.1.2", "torchvision==0.16.2", "torchaudio==2.1.2"),
    ]
    for torch_pkg, vision_pkg, audio_pkg in CANDIDATES:
        try:
            print(f"[setup] installing {torch_pkg}+{vision_pkg}+{audio_pkg} "
                  f"(cu121, sm_60/P100 support)...", flush=True)
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--quiet",
                 torch_pkg, vision_pkg, audio_pkg,
                 "--index-url", "https://download.pytorch.org/whl/cu121"],
                check=True, timeout=900,
            )
            break
        except Exception as e:
            print(f"[setup] {torch_pkg} install failed ({e}), trying next candidate", flush=True)
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet",
             "transformers>=4.45,<4.50", "accelerate>=0.33", "safetensors"],
            check=True, timeout=600,
        )
    except Exception as e:
        print(f"[setup] transformers/accelerate install failed: {e}", flush=True)

import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pydicom  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402
from transformers import AutoModel  # noqa: E402

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]
PLANE_ORDER = {"Sagittal": 0, "Coronal": 1, "Axial": 2}
NUM_LABELS = len(LABELS)

DINOV2_MODEL = os.environ.get("KNEE_DINOV2_MODEL", "facebook/dinov2-small")
IMG_SIZE = 224          # DINOv2 patch size 14 -> 224 = 16x16 patches
MAX_SLICES = 8           # ViT forward is far more expensive per-slice than v1-v4's CNN
K_SERIES = 3
UNFREEZE_LAST = 6
LR_BACKBONE = 8e-6
LR_HEAD = 1e-3
RUN_NAME = "lossweighted_15ep"
EPOCHS = int(os.environ.get("KNEE_EPOCHS", "15"))      # v8's weighting + v7-moreepochs' epoch count
VAL_FRAC = 0.2
SEED = 42
MAX_EXTRA = int(os.environ.get("KNEE_MAX_EXTRA", "1500"))  # v5's settings, same reasoning
LAT_DEAD_ZONE_MM = 20.0
TIME_BUDGET_S = float(os.environ.get("KNEE_TIME_BUDGET_S", str(7 * 3600)))  # safety margin under Kaggle's session limit

# The core idea being tested in v8: a gold-labeled study's loss should count
# for more than an extracted-label study's, since only ~76% of extracted
# labels are correct (results/extract_labels_v1/validation_report.json) --
# right now every study contributes equally regardless of label quality.
GOLD_LOSS_WEIGHT = float(os.environ.get("KNEE_GOLD_WEIGHT", "1.0"))
EXTRACTED_LOSS_WEIGHT = float(os.environ.get("KNEE_EXTRACTED_WEIGHT", "0.3"))

OUT_DIR = os.environ.get("KNEE_OUT_DIR", "/kaggle/working")
CACHE_DIR = os.environ.get("KNEE_CACHE_DIR", os.path.join(tempfile.gettempdir(), f"knee_tensor_cache_v9_{RUN_NAME}"))
MIN_FREE_GB_FOR_CACHE = 3.0


def find_input_dir():
    env = os.environ.get("KNEE_INPUT_DIR")
    if env and os.path.exists(os.path.join(env, "train.csv")):
        return env
    candidates = [
        "/kaggle/input/rsna-knee-abnormality-detection",
        "/kaggle/input/competitions/rsna-knee-abnormality-detection",
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "train.csv")):
            return c
    hits = glob.glob("/kaggle/input/**/train.csv", recursive=True)
    if hits:
        return os.path.dirname(hits[0])
    raise RuntimeError("could not locate train.csv under /kaggle/input")


def find_extracted_csv():
    env = os.environ.get("KNEE_EXTRACTED_CSV")
    if env and os.path.exists(env):
        return env
    hits = glob.glob("/kaggle/input/**/report_labels_extracted.csv", recursive=True)
    if hits:
        return hits[0]
    raise RuntimeError(
        "could not locate report_labels_extracted.csv -- add "
        "liamschneider0907/rsna-knee-extract-labels-v1-full to this kernel's "
        "kernel_sources so its output is mounted under /kaggle/input"
    )


# ------------------------------------------------------------- geometry/IO
def _resize(arr, size):
    from PIL import Image
    return np.asarray(Image.fromarray(arr.astype(np.float32))
                       .resize((size, size), Image.BILINEAR), dtype=np.float32)


def _pad_to_square(arr):
    h, w = arr.shape
    if h == w:
        return arr
    s = max(h, w)
    out = np.zeros((s, s), dtype=arr.dtype)
    y0, x0 = (s - h) // 2, (s - w) // 2
    out[y0:y0 + h, x0:x0 + w] = arr
    return out


def _slice_order_key(ds):
    """Signed through-plane position from patient-space geometry, or None.

    Projecting ImagePositionPatient onto the slice normal (cross product of
    the row/column direction cosines) gives a value that increases
    monotonically along the true physical stack -- unlike file name (an
    opaque UID) or InstanceNumber (not contractually tied to spatial order).
    """
    try:
        iop = np.asarray(ds.ImageOrientationPatient, dtype=float)
        ipp = np.asarray(ds.ImagePositionPatient, dtype=float)
        if len(iop) < 6 or len(ipp) < 3:
            return None
        normal = np.cross(iop[:3], iop[3:6])
        return float(np.dot(ipp, normal))
    except Exception:
        return None


def _study_laterality(series_dirs):
    """'L' / 'R' / None for a study, from DICOM tags first, geometry second.

    series_dirs: list of directories, one per series, to probe (one header
    read each, cheap since stop_before_pixels skips the pixel data).
    """
    tag_votes = []
    xs = []
    for d in series_dirs:
        files = [f for f in os.listdir(d) if f.endswith(".dcm")]
        if not files:
            continue
        try:
            ds = pydicom.dcmread(os.path.join(d, files[0]), stop_before_pixels=True)
        except Exception:
            continue
        for tag in ("Laterality", "ImageLaterality"):
            v = getattr(ds, tag, None)
            if v:
                v = str(v).strip().upper()[:1]
                if v in ("L", "R"):
                    tag_votes.append(v)
        try:
            iop = np.asarray(ds.ImageOrientationPatient, dtype=float)
            ipp = np.asarray(ds.ImagePositionPatient, dtype=float)
            cols = float(getattr(ds, "Columns", 0) or 0)
            rows = float(getattr(ds, "Rows", 0) or 0)
            ps = getattr(ds, "PixelSpacing", None)
            if len(iop) >= 6 and len(ipp) >= 3 and ps and cols and rows:
                px, py = float(ps[0]), float(ps[1])
                # centre of the image, not the corner ImagePositionPatient
                # itself names -- the corner is offset by half a field of
                # view, enough to flip the sign for a knee near the midline.
                c = ipp[:3] + iop[:3] * py * cols / 2 + iop[3:6] * px * rows / 2
                xs.append(float(c[0]))
        except Exception:
            pass
    if tag_votes:
        return max(set(tag_votes), key=tag_votes.count)
    if xs:
        m = float(np.median(xs))
        if abs(m) >= LAT_DEAD_ZONE_MM:
            # DICOM patient coordinates are LPS: +x is the patient's left.
            return "R" if m < 0 else "L"
    return None


def read_series(series_dir, plane, laterality, img_size=IMG_SIZE, max_slices=MAX_SLICES):
    files = [os.path.join(series_dir, f) for f in os.listdir(series_dir) if f.endswith(".dcm")]
    if not files:
        return None
    slices = []
    for f in files:
        try:
            ds = pydicom.dcmread(f)
            arr = ds.pixel_array.astype(np.float32)
        except Exception:
            continue
        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        arr = arr * slope + intercept
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            arr = arr.max() - arr
        geo_key = _slice_order_key(ds)
        if geo_key is not None:
            order_key = (0, geo_key)
        else:
            inst = getattr(ds, "InstanceNumber", None)
            order_key = (1, int(inst) if inst is not None else 0)
        slices.append((order_key, arr))
    if not slices:
        return None
    slices.sort(key=lambda x: x[0])
    vol = np.stack([a for _, a in slices])

    lo, hi = np.percentile(vol, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    vol = np.clip((vol - lo) / (hi - lo), 0, 1)

    if laterality == "R":
        if plane in ("Coronal", "Axial"):
            vol = vol[:, :, ::-1].copy()
        elif plane == "Sagittal":
            vol = vol[::-1].copy()

    if vol.shape[0] > max_slices:
        start = (vol.shape[0] - max_slices) // 2
        vol = vol[start:start + max_slices]
    return np.stack([_resize(_pad_to_square(s), img_size) for s in vol])


def pick_series(series_df, k=K_SERIES):
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


def train_val_split(df, val_frac=VAL_FRAC, seed=SEED):
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_val = max(1, int(round(len(shuffled) * val_frac)))
    return shuffled.iloc[n_val:].reset_index(drop=True), shuffled.iloc[:n_val].reset_index(drop=True)


# --------------------------------------------------------------------- data
class KneeStudyDataset(Dataset):
    def __init__(self, studies_df, series_df, series_root, cache_dir):
        self.studies = studies_df.reset_index(drop=True)
        self.series_df = series_df
        self.series_root = series_root
        self.cache_dir = cache_dir
        self.cache_disabled = False
        os.makedirs(cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.studies)

    def _decode(self, study_uid):
        study_series = self.series_df[self.series_df["StudyInstanceUID"] == study_uid]
        series_uids = pick_series(study_series)
        series_dirs = [os.path.join(self.series_root, study_uid, s) for s in series_uids]
        laterality = _study_laterality([d for d in series_dirs if os.path.isdir(d)])

        arrays = []
        for suid, series_dir in zip(series_uids, series_dirs):
            if not os.path.isdir(series_dir):
                continue
            plane_rows = study_series.loc[study_series["SeriesInstanceUID"] == suid, "Anatomical_Plane"]
            plane = plane_rows.iloc[0] if len(plane_rows) else "Sagittal"
            vol = read_series(series_dir, plane=plane, laterality=laterality)
            if vol is not None:
                arrays.append(vol)
        return arrays

    def __getitem__(self, idx):
        row = self.studies.iloc[idx]
        study_uid = row["StudyInstanceUID"]
        label_vals = pd.to_numeric(row[LABELS], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        label = torch.from_numpy(label_vals)
        # "source" is only present on the combined training set (gold vs
        # extracted-label studies) -- val/test sets are gold-only or
        # unlabeled, so a study without the column just isn't down-weighted.
        weight = float(row["loss_weight"]) if "loss_weight" in row else 1.0

        cache_path = os.path.join(self.cache_dir, f"{study_uid}.npz")
        if os.path.exists(cache_path):
            with np.load(cache_path) as npz:
                arrays = [npz[k] for k in sorted(npz.files)]
        else:
            arrays = self._decode(study_uid)
            if not self.cache_disabled:
                try:
                    free_gb = shutil.disk_usage(self.cache_dir).free / 1024 ** 3
                    if free_gb < MIN_FREE_GB_FOR_CACHE:
                        self.cache_disabled = True
                        print(f"[cache] only {free_gb:.1f} GB free, disabling further "
                              f"cache writes (falling back to re-decode every epoch)", flush=True)
                    else:
                        np.savez_compressed(cache_path, **{f"s{i}": a for i, a in enumerate(arrays)})
                except OSError as e:
                    self.cache_disabled = True
                    print(f"[cache] write failed ({e}), disabling further cache writes", flush=True)

        tensors = [torch.from_numpy(a).unsqueeze(1) for a in arrays]
        return tensors, label, study_uid, weight


# -------------------------------------------------------------------- model
class DinoSliceEncoder(nn.Module):
    """DINOv2 backbone -> per-slice (CLS + patch-mean) embedding.

    Only the last UNFREEZE_LAST transformer blocks and the final layernorm
    are trainable; everything earlier stays frozen, since there isn't
    remotely enough data here to safely adapt a whole ViT from scratch.
    """

    def __init__(self, model_name=DINOV2_MODEL, unfreeze_last=UNFREEZE_LAST):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size
        self.out_dim = hidden * 2

        n_layers = len(self.backbone.encoder.layer)
        for p in self.backbone.parameters():
            p.requires_grad = False
        for blk in self.backbone.encoder.layer[max(0, n_layers - unfreeze_last):]:
            for p in blk.parameters():
                p.requires_grad = True
        if hasattr(self.backbone, "layernorm") and self.backbone.layernorm is not None:
            for p in self.backbone.layernorm.parameters():
                p.requires_grad = True

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def trainable_parameters(self):
        return [p for p in self.backbone.parameters() if p.requires_grad]

    def forward(self, x):
        # x: (S, 1, H, W) in [0, 1] -> replicate to 3 channels, ImageNet-normalize
        x = x.repeat(1, 3, 1, 1)
        x = (x - self.mean) / self.std
        out = self.backbone(pixel_values=x).last_hidden_state
        cls = out[:, 0]
        patch_mean = out[:, 1:].mean(1)
        return torch.cat([cls, patch_mean], dim=1)


class KneeMILModel(nn.Module):
    def __init__(self, num_labels=NUM_LABELS):
        super().__init__()
        self.encoder = DinoSliceEncoder()
        self.head = nn.Linear(self.encoder.out_dim, num_labels)

    def forward_series(self, slices):
        embeds = self.encoder(slices)
        pooled, _ = embeds.max(dim=0)
        return self.head(pooled)

    def forward_study(self, series_list):
        logits = torch.stack([self.forward_series(s) for s in series_list])
        return logits.mean(dim=0)

    def forward_batch(self, studies):
        return torch.stack([self.forward_study(s) for s in studies])


# ---------------------------------------------------------------- eval loop
def evaluate(model, ds, device):
    model.eval()
    probs_all, labels_all = [], []
    with torch.no_grad():
        for i in range(len(ds)):
            series_list, label, uid, _ = ds[i]
            if not series_list:
                continue
            series_list = [s.to(device) for s in series_list]
            logits = model.forward_batch([series_list])
            probs_all.append(torch.sigmoid(logits).cpu().numpy()[0])
            labels_all.append(label.numpy())

    if not probs_all:
        return None, 0, {}

    probs = np.stack(probs_all)
    labels = np.stack(labels_all)
    per_label = {}
    for j, name in enumerate(LABELS):
        y = labels[:, j]
        if len(np.unique(y)) < 2:
            continue
        per_label[name] = roc_auc_score(y, probs[:, j])

    if not per_label:
        return None, 0, {}
    return float(np.mean(list(per_label.values()))), len(per_label), per_label


# --------------------------------------------------------------------- main
def pick_device():
    if not torch.cuda.is_available():
        return "cpu"
    try:
        x = torch.randn(2, 1, 8, 8, device="cuda")
        w = torch.randn(2, 1, 3, 3, device="cuda")
        torch.nn.functional.conv2d(x, w)
        torch.cuda.synchronize()
        return "cuda"
    except Exception as e:
        print(f"[setup] CUDA smoke test failed ({e}) -> falling back to CPU", flush=True)
        return "cpu"


def main():
    t0 = time.time()
    input_dir = find_input_dir()
    device = pick_device()
    print(f"[setup] torch={torch.__version__} device={device} epochs={EPOCHS} "
          f"max_extra={MAX_EXTRA} model={DINOV2_MODEL} cache_dir={CACHE_DIR} "
          f"gold_weight={GOLD_LOSS_WEIGHT} extracted_weight={EXTRACTED_LOSS_WEIGHT}", flush=True)

    train = pd.read_csv(os.path.join(input_dir, "train.csv"), encoding="latin-1")
    train_series = pd.read_csv(os.path.join(input_dir, "train_series.csv"))
    gold = train.dropna(subset=LABELS, how="all").reset_index(drop=True)
    gold_train_df, val_df = train_val_split(gold)

    extracted_path = find_extracted_csv()
    extracted = pd.read_csv(extracted_path)
    extracted = extracted[~extracted["parse_failed"]].reset_index(drop=True)
    if len(extracted) > MAX_EXTRA:
        extracted = extracted.sample(n=MAX_EXTRA, random_state=SEED).reset_index(drop=True)
    print(f"[data] gold={len(gold)} ({len(gold_train_df)} train / {len(val_df)} val), "
          f"extracted usable={len(extracted)} (capped at {MAX_EXTRA})", flush=True)

    gold_train_df = gold_train_df.copy()
    gold_train_df["loss_weight"] = GOLD_LOSS_WEIGHT
    extracted = extracted.copy()
    extracted["loss_weight"] = EXTRACTED_LOSS_WEIGHT

    extra_cols = ["StudyInstanceUID", "loss_weight"] + LABELS
    train_df = pd.concat([gold_train_df[extra_cols], extracted[extra_cols]], ignore_index=True)
    print(f"[data] total training studies: {len(train_df)}", flush=True)

    series_root = os.path.join(input_dir, "train_series")
    train_ds = KneeStudyDataset(train_df, train_series, series_root,
                                 cache_dir=os.path.join(CACHE_DIR, "train"))
    val_ds = KneeStudyDataset(val_df, train_series, series_root,
                               cache_dir=os.path.join(CACHE_DIR, "val"))

    print("[setup] loading DINOv2 backbone (first run downloads weights)...", flush=True)
    model = KneeMILModel().to(device)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[setup] model loaded: {n_trainable/1e6:.1f}M / {n_total/1e6:.1f}M params trainable, "
          f"elapsed={time.time()-t0:.0f}s", flush=True)

    opt = torch.optim.AdamW([
        {"params": model.encoder.trainable_parameters(), "lr": LR_BACKBONE},
        {"params": model.head.parameters(), "lr": LR_HEAD},
    ])
    loss_fn = nn.BCEWithLogitsLoss()

    os.makedirs(OUT_DIR, exist_ok=True)
    ckpt_path = os.path.join(OUT_DIR, f"model_v9_{RUN_NAME}.pt")
    history_path = os.path.join(OUT_DIR, f"history_v9_{RUN_NAME}.json")

    def save_history(history, best_auc, best_epoch, final_per_label, stopped_early):
        # History is written every epoch regardless of the checkpoint --
        # cheap, and it's the full curve, not just the best point.
        with open(history_path, "w") as f:
            json.dump({"history": history, "best_val_macro_auc": best_auc,
                        "best_epoch": best_epoch, "final_per_label_auc": final_per_label,
                        "n_train": len(train_df), "n_val": len(val_df),
                        "n_extracted_used": len(extracted), "max_extra": MAX_EXTRA,
                        "dinov2_model": DINOV2_MODEL, "unfreeze_last": UNFREEZE_LAST,
                        "torch_version": torch.__version__, "device": device,
                        "run_name": RUN_NAME, "epochs_configured": EPOCHS,
                        "gold_loss_weight": GOLD_LOSS_WEIGHT,
                        "extracted_loss_weight": EXTRACTED_LOSS_WEIGHT,
                        "stopped_early_time_budget": stopped_early}, f, indent=2)

    def save_checkpoint(state_dict):
        torch.save({"model": state_dict, "img_size": IMG_SIZE,
                    "max_slices": MAX_SLICES, "k_series": K_SERIES,
                    "dinov2_model": DINOV2_MODEL}, ckpt_path)

    history = []
    best_auc, best_epoch = -1.0, -1
    best_state = None
    stopped_early = False
    for epoch in range(EPOCHS):
        if time.time() - t0 > TIME_BUDGET_S:
            print(f"[time-budget] {TIME_BUDGET_S:.0f}s reached before epoch {epoch+1} "
                  f"started -- stopping cleanly with {epoch} epoch(s) done instead of "
                  f"risking a mid-epoch kill", flush=True)
            stopped_early = True
            break

        model.train()
        epoch_losses = []
        for i in range(len(train_ds)):
            try:
                series_list, label, uid, weight = train_ds[i]
                if not series_list:
                    continue
                series_list = [s.to(device) for s in series_list]
                label = label.unsqueeze(0).to(device)

                opt.zero_grad()
                logits = model.forward_batch([series_list])
                raw_loss = loss_fn(logits, label)
                (raw_loss * weight).backward()
                opt.step()
                # Logged unweighted, so train_loss stays comparable to v5's
                # plot/number -- the weighting only changes what backward()
                # sees, not what gets reported here.
                epoch_losses.append(raw_loss.item())
            except Exception as e:
                print(f"  [warn] study {i} failed ({type(e).__name__}: {e}), skipping", flush=True)
                continue

            if (i + 1) % 200 == 0:
                print(f"  [epoch {epoch+1}] {i+1}/{len(train_ds)} studies, "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)

        val_auc, n_scored, per_label = evaluate(model, val_ds, device)
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        history.append({"epoch": epoch + 1, "train_loss": train_loss,
                         "val_macro_auc": val_auc, "n_labels_scored": n_scored})
        if val_auc is not None and val_auc > best_auc:
            best_auc, best_epoch = val_auc, epoch + 1
            # A CPU copy, not a reference -- the GPU-resident state keeps
            # changing every subsequent epoch, so without cloning this
            # would just end up holding the final epoch's weights again.
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        auc_str = f"{val_auc:.4f}" if val_auc is not None else "n/a"
        print(f"[epoch {epoch+1}/{EPOCHS}] train_loss={train_loss:.4f} "
              f"val_macro_auc={auc_str} (n_labels={n_scored}/{NUM_LABELS}) "
              f"elapsed={time.time()-t0:.0f}s", flush=True)

        # History after every epoch, not just at the end -- if this run
        # dies unexpectedly (time budget aside, e.g. a Kaggle infra hiccup),
        # whatever epochs completed are still recoverable kernel output.
        # The checkpoint only overwrites when there's an actual new best,
        # so it always reflects the best-scoring epoch, not the latest one.
        save_history(history, best_auc, best_epoch, per_label, stopped_early)
        if best_state is not None and best_epoch == epoch + 1:
            save_checkpoint(best_state)

    if best_state is None:
        # val AUC never scored a single epoch (e.g. every label came back
        # constant in this particular 12-study validation split) -- fall
        # back to whatever the model holds rather than write no checkpoint.
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        save_checkpoint(best_state)
    print(f"[done] saved {ckpt_path} (best epoch {best_epoch})", flush=True)
    print(f"[done] best val_macro_auc={best_auc:.4f} at epoch {best_epoch}", flush=True)

    # Evaluate (and generate the submission below) with the best-epoch
    # weights, not whatever the model happens to hold after the last
    # epoch trained -- v6 saved the latter by coincidence of it also
    # being the best epoch; nothing here should keep relying on that.
    if best_state is not None:
        model.load_state_dict(best_state)
    final_val_auc, final_n, final_per_label = evaluate(model, val_ds, device)
    save_history(history, best_auc, best_epoch, final_per_label, stopped_early)
    print(f"[done] final per-label val AUC: {final_per_label}", flush=True)

    test_csv = os.path.join(input_dir, "test.csv")
    test_series_csv = os.path.join(input_dir, "test_series.csv")
    if os.path.exists(test_csv):
        test = pd.read_csv(test_csv)
        test_series = pd.read_csv(test_series_csv)
        test_ds = KneeStudyDataset(test.assign(**{lab: 0.0 for lab in LABELS}), test_series,
                                    os.path.join(input_dir, "test_series"),
                                    cache_dir=os.path.join(CACHE_DIR, "test"))
        model.eval()
        rows = []
        with torch.no_grad():
            for i in range(len(test_ds)):
                series_list, _, uid, _ = test_ds[i]
                row = {"StudyInstanceUID": uid}
                if series_list:
                    series_list = [s.to(device) for s in series_list]
                    logits = model.forward_batch([series_list])
                    probs = torch.sigmoid(logits).cpu().numpy()[0]
                else:
                    probs = np.full(NUM_LABELS, 0.5)
                for j, c in enumerate(LABELS):
                    row[c] = float(probs[j])
                rows.append(row)
        sub_path = os.path.join(OUT_DIR, "submission.csv")
        pd.DataFrame(rows, columns=["StudyInstanceUID"] + LABELS).to_csv(sub_path, index=False)
        print(f"[done] wrote {sub_path} ({len(rows)} studies)", flush=True)

    print(f"[done] total wall time {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
