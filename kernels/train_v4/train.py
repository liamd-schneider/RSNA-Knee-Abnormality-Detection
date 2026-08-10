"""RSNA Knee Abnormality Detection -- baseline training kernel (v4).

Same model as v3, but trains on the 46 gold-labeled training studies PLUS
report-derived labels from kernels/extract_labels_v1_full (validated at
75.9% mean accuracy / 0.64 mean F1 against the 58 gold studies before ever
being trusted here -- see results/extract_labels_v1/validation_report.json).
Validation still comes ONLY from the 12 held-out gold studies: extracted
labels are noisy enough that they must never be treated as ground truth,
only as extra (weaker) training signal.

Two things had to change from v3 because the dataset is now ~95x bigger
(up to ~4,349 extra studies vs. 46):

1. Disk-backed tensor cache instead of v3's in-memory dict cache -- holding
   every study's decoded tensors in RAM doesn't fit anymore (back-of-envelope:
   4,349 studies x 3 series x 24 slices x 160x160 float32 is on the order of
   30+ GB). Each study's tensors are decoded once and saved to a local .npz
   file; every later read (this epoch or a future one) is a cheap disk read
   instead of a DICOM re-decode.

2. A cap on how many extracted-label studies are actually used
   (KNEE_MAX_EXTRA, default 1500), not all ~4,349. Decoding a DICOM series
   cold costs ~2.4s (measured in kernels/train_v3's dry run) -- decoding all
   4,349 once would be ~3h of wall time on its own, before any training
   happens, real risk of blowing past a single Kaggle session. 1500 is a
   deliberate, adjustable trade-off: a large jump over 46 studies while
   keeping the unavoidable first-pass decode cost within one session.

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

# ---------------------------------------------------------- torch bootstrap
if not os.environ.get("KNEE_SKIP_TORCH_INSTALL"):
    for candidate in ["torch==2.3.1", "torch==2.1.2"]:
        try:
            print(f"[setup] installing {candidate} (cu121, sm_60/P100 support)...", flush=True)
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--quiet", candidate,
                 "--index-url", "https://download.pytorch.org/whl/cu121"],
                check=True, timeout=600,
            )
            break
        except Exception as e:
            print(f"[setup] {candidate} install failed ({e}), trying next candidate", flush=True)

import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pydicom  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
          "Contusion", "Fracture"]
PLANE_ORDER = {"Sagittal": 0, "Coronal": 1, "Axial": 2}
NUM_LABELS = len(LABELS)

IMG_SIZE = 160
MAX_SLICES = 24
K_SERIES = 3
EPOCHS = int(os.environ.get("KNEE_EPOCHS", "8"))
LR = 1e-3
VAL_FRAC = 0.2
SEED = 42
MAX_EXTRA = int(os.environ.get("KNEE_MAX_EXTRA", "1500"))

OUT_DIR = os.environ.get("KNEE_OUT_DIR", "/kaggle/working")
# Deliberately NOT under OUT_DIR: /kaggle/working is uploaded as the kernel's
# output when the run ends, and the cache (thousands of .npz files) has no
# reason to be part of that -- found out the hard way when downloading a
# failed run's output pulled down 250+ MB of cache files before it got to
# the one log worth reading.
#
# tempfile.gettempdir() rather than a guessed "/kaggle/..." path: this run
# died silently (no traceback -- consistent with an out-of-space/OOM kill,
# not a Python exception) after processing a few hundred studies' worth of
# cache writes, at a slightly different study count each time, which points
# at resource exhaustion rather than a specific bad input. Guessing at an
# unconfirmed Kaggle-specific directory was exactly the kind of assumption
# that causes this -- /tmp always exists and is never guessed at.
import tempfile  # noqa: E402
CACHE_DIR = os.environ.get("KNEE_CACHE_DIR", os.path.join(tempfile.gettempdir(), "knee_tensor_cache"))
MIN_FREE_GB_FOR_CACHE = 3.0  # below this, stop writing new cache entries rather than risk the run


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


# ------------------------------------------------------------------ dicom IO
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


def read_series(series_dir, img_size=IMG_SIZE, max_slices=MAX_SLICES):
    files = [os.path.join(series_dir, f) for f in os.listdir(series_dir) if f.endswith(".dcm")]
    if not files:
        return None
    slices = []
    for f in files:
        try:
            ds = pydicom.dcmread(f)
            arr = ds.pixel_array.astype(np.float32)
        except Exception:
            # A handful of DICOMs in this corpus have truncated/corrupted
            # pixel data -- drop the bad slice rather than crash the whole
            # run over it (found the hard way: this killed a training run
            # ~40 min in, at study 800/1546).
            continue
        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        arr = arr * slope + intercept
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            arr = arr.max() - arr
        inst = getattr(ds, "InstanceNumber", None)
        slices.append((int(inst) if inst is not None else 0, arr))
    if not slices:
        return None
    slices.sort(key=lambda x: x[0])
    vol = np.stack([a for _, a in slices])
    lo, hi = np.percentile(vol, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    vol = np.clip((vol - lo) / (hi - lo), 0, 1)
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
    """Disk-cached: each study's decoded tensors are saved to CACHE_DIR on
    first read and loaded from there afterwards, instead of held in RAM
    (v3's approach) or re-decoded from DICOM every epoch (v1/v2's)."""

    def __init__(self, studies_df, series_df, series_root, cache_dir):
        self.studies = studies_df.reset_index(drop=True)
        self.series_df = series_df
        self.series_root = series_root
        self.cache_dir = cache_dir
        self.cache_disabled = False  # flips permanently once disk space runs low
        os.makedirs(cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.studies)

    def _decode(self, study_uid):
        study_series = self.series_df[self.series_df["StudyInstanceUID"] == study_uid]
        series_uids = pick_series(study_series)
        arrays = []
        for suid in series_uids:
            series_dir = os.path.join(self.series_root, study_uid, suid)
            if not os.path.isdir(series_dir):
                continue
            vol = read_series(series_dir)
            if vol is not None:
                arrays.append(vol)
        return arrays

    def __getitem__(self, idx):
        row = self.studies.iloc[idx]
        study_uid = row["StudyInstanceUID"]
        label_vals = pd.to_numeric(row[LABELS], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        label = torch.from_numpy(label_vals)

        cache_path = os.path.join(self.cache_dir, f"{study_uid}.npz")
        if os.path.exists(cache_path):
            with np.load(cache_path) as npz:
                arrays = [npz[k] for k in sorted(npz.files)]
        else:
            arrays = self._decode(study_uid)
            # Caching is a speed optimization, not a correctness requirement --
            # a write that fails (disk full) or that would push free space
            # below the safety margin must never take the training run down
            # with it. Once disabled it stays disabled for the rest of this
            # dataset's lifetime rather than re-checking every single item.
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
        return tensors, label, study_uid


# -------------------------------------------------------------------- model
class SliceEncoder(nn.Module):
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
        return self.net(x).flatten(1)


class KneeMILModel(nn.Module):
    def __init__(self, num_labels=NUM_LABELS, embed_dim=128):
        super().__init__()
        self.encoder = SliceEncoder(embed_dim)
        self.head = nn.Linear(embed_dim, num_labels)

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
            series_list, label, uid = ds[i]
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
          f"max_extra={MAX_EXTRA} cache_dir={CACHE_DIR}", flush=True)

    train = pd.read_csv(os.path.join(input_dir, "train.csv"), encoding="latin-1")
    train_series = pd.read_csv(os.path.join(input_dir, "train_series.csv"))
    gold = train.dropna(subset=LABELS, how="all").reset_index(drop=True)
    gold_train_df, val_df = train_val_split(gold)  # val ALWAYS from real gold labels only

    extracted_path = find_extracted_csv()
    extracted = pd.read_csv(extracted_path)
    extracted = extracted[~extracted["parse_failed"]].reset_index(drop=True)
    if len(extracted) > MAX_EXTRA:
        extracted = extracted.sample(n=MAX_EXTRA, random_state=SEED).reset_index(drop=True)
    print(f"[data] gold={len(gold)} ({len(gold_train_df)} train / {len(val_df)} val), "
          f"extracted usable={len(extracted)} (capped at {MAX_EXTRA})", flush=True)

    extra_cols = ["StudyInstanceUID"] + LABELS
    train_df = pd.concat([gold_train_df[extra_cols], extracted[extra_cols]], ignore_index=True)
    print(f"[data] total training studies: {len(train_df)}", flush=True)

    series_root = os.path.join(input_dir, "train_series")
    train_ds = KneeStudyDataset(train_df, train_series, series_root,
                                 cache_dir=os.path.join(CACHE_DIR, "train"))
    val_ds = KneeStudyDataset(val_df, train_series, series_root,
                               cache_dir=os.path.join(CACHE_DIR, "val"))

    model = KneeMILModel().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()

    history = []
    best_auc, best_epoch = -1.0, -1
    for epoch in range(EPOCHS):
        model.train()
        epoch_losses = []
        for i in range(len(train_ds)):
            try:
                series_list, label, uid = train_ds[i]
                if not series_list:
                    continue
                series_list = [s.to(device) for s in series_list]
                label = label.unsqueeze(0).to(device)

                opt.zero_grad()
                logits = model.forward_batch([series_list])
                loss = loss_fn(logits, label)
                loss.backward()
                opt.step()
                epoch_losses.append(loss.item())
            except Exception as e:
                # One bad study (decode error the DICOM-level guards didn't
                # catch, a one-off CUDA hiccup, ...) must not take the whole
                # multi-hour run down with it -- skip it and keep going.
                print(f"  [warn] study {i} failed ({type(e).__name__}: {e}), skipping", flush=True)
                continue

            if (i + 1) % 200 == 0:
                print(f"  [epoch {epoch+1}] {i+1}/{len(train_ds)} studies, "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)

        val_auc, n_scored, per_label = evaluate(model, val_ds, device)
        train_loss = float(np.mean(epoch_losses))
        history.append({"epoch": epoch + 1, "train_loss": train_loss,
                         "val_macro_auc": val_auc, "n_labels_scored": n_scored})
        if val_auc is not None and val_auc > best_auc:
            best_auc, best_epoch = val_auc, epoch + 1

        auc_str = f"{val_auc:.4f}" if val_auc is not None else "n/a"
        print(f"[epoch {epoch+1}/{EPOCHS}] train_loss={train_loss:.4f} "
              f"val_macro_auc={auc_str} (n_labels={n_scored}/{NUM_LABELS}) "
              f"elapsed={time.time()-t0:.0f}s", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    ckpt_path = os.path.join(OUT_DIR, "model_v4.pt")
    torch.save({"model": model.state_dict(), "img_size": IMG_SIZE,
                "max_slices": MAX_SLICES, "k_series": K_SERIES}, ckpt_path)
    print(f"[done] saved {ckpt_path}", flush=True)
    print(f"[done] best val_macro_auc={best_auc:.4f} at epoch {best_epoch}", flush=True)

    final_val_auc, final_n, final_per_label = evaluate(model, val_ds, device)
    with open(os.path.join(OUT_DIR, "history_v4.json"), "w") as f:
        json.dump({"history": history, "best_val_macro_auc": best_auc,
                    "best_epoch": best_epoch, "final_per_label_auc": final_per_label,
                    "n_train": len(train_df), "n_val": len(val_df),
                    "n_extracted_used": len(extracted), "max_extra": MAX_EXTRA,
                    "torch_version": torch.__version__, "device": device}, f, indent=2)
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
                series_list, _, uid = test_ds[i]
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
