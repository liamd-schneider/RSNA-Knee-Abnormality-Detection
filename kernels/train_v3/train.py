"""RSNA Knee Abnormality Detection -- baseline training kernel (v3).

Same model/data/validation as v2 (kernels/train_v2), plus two speed fixes
that came out of watching v2 run at ~90s/epoch on CPU:

1. In-memory tensor caching -- v1/v2 re-read and re-decoded every study's
   DICOMs from scratch on every single epoch, even though the images never
   change between epochs. That I/O + decode work is almost certainly the
   actual bottleneck, not the (tiny) model's forward/backward pass. Now every
   study is decoded once and cached in memory for the rest of the run.

2. A real GPU, not just an allocated one -- v1/v2 requested a GPU
   (enable_gpu=true) but Kaggle assigned an older P100 whose compute
   capability (sm_60) isn't supported by the CUDA kernels shipped in the
   pre-installed PyTorch build, so every run silently fell back to CPU while
   *still burning P100 quota hours* for the privilege. This script installs
   an older torch+cu121 build (still ships sm_60 kernels) before torch is
   ever imported, so training can actually run on the GPU it was allocated.
   Needs internet enabled for this kernel specifically -- fine for a
   dev/training run, never for the final no-internet submission notebook.

Local dry run:
    KNEE_INPUT_DIR=data/dryrun KNEE_OUT_DIR=/tmp/out KNEE_EPOCHS=1 \
    KNEE_SKIP_TORCH_INSTALL=1 python train.py
"""
import os
import subprocess
import sys

# ---------------------------------------------------------- torch bootstrap
# Must run before `import torch` anywhere -- installing into an
# already-imported torch process doesn't actually swap the loaded extension.
if not os.environ.get("KNEE_SKIP_TORCH_INSTALL"):
    for candidate in ["torch==2.3.1", "torch==2.1.2"]:
        try:
            print(f"[setup] installing {candidate} (cu121, includes sm_60/P100 kernels)...", flush=True)
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--quiet", candidate,
                 "--index-url", "https://download.pytorch.org/whl/cu121"],
                check=True, timeout=600,
            )
            break
        except Exception as e:
            print(f"[setup] {candidate} install failed ({e}), trying next candidate", flush=True)

import glob  # noqa: E402
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
EPOCHS = int(os.environ.get("KNEE_EPOCHS", "20"))
LR = 1e-3
VAL_FRAC = 0.2
SEED = 42

OUT_DIR = os.environ.get("KNEE_OUT_DIR", "/kaggle/working")


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
        ds = pydicom.dcmread(f)
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        arr = arr * slope + intercept
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            arr = arr.max() - arr
        inst = getattr(ds, "InstanceNumber", None)
        slices.append((int(inst) if inst is not None else 0, arr))
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
    def __init__(self, studies_df, series_df, series_root, has_labels=True, cache=True):
        self.studies = studies_df.reset_index(drop=True)
        self.series_df = series_df
        self.series_root = series_root
        self.has_labels = has_labels
        self._cache = {} if cache else None

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, idx):
        if self._cache is not None and idx in self._cache:
            return self._cache[idx]

        row = self.studies.iloc[idx]
        study_uid = row["StudyInstanceUID"]
        study_series = self.series_df[self.series_df["StudyInstanceUID"] == study_uid]
        series_uids = pick_series(study_series)

        tensors = []
        for suid in series_uids:
            series_dir = os.path.join(self.series_root, study_uid, suid)
            if not os.path.isdir(series_dir):
                continue
            vol = read_series(series_dir)
            if vol is None:
                continue
            tensors.append(torch.from_numpy(vol).unsqueeze(1))

        if self.has_labels:
            label_vals = pd.to_numeric(row[LABELS], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            label = torch.from_numpy(label_vals)
            result = (tensors, label, study_uid)
        else:
            result = (tensors, study_uid)

        if self._cache is not None:
            self._cache[idx] = result
        return result


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
    """Returns (macro_auc or None, n_labels_scored, per_label_auc dict)."""
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
    print(f"[setup] torch={torch.__version__} cuda_available={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"[setup] gpu={torch.cuda.get_device_name(0)} "
              f"cap={torch.cuda.get_device_capability(0)}", flush=True)
    device = pick_device()
    print(f"[setup] input_dir={input_dir} device={device} epochs={EPOCHS} "
          f"img_size={IMG_SIZE} k_series={K_SERIES} val_frac={VAL_FRAC}", flush=True)

    train = pd.read_csv(os.path.join(input_dir, "train.csv"))
    train_series = pd.read_csv(os.path.join(input_dir, "train_series.csv"))
    labeled = train.dropna(subset=LABELS, how="all").reset_index(drop=True)
    train_df, val_df = train_val_split(labeled)
    print(f"[data] {len(train)} studies total, {len(labeled)} labeled -> "
          f"{len(train_df)} train / {len(val_df)} val", flush=True)

    series_root = os.path.join(input_dir, "train_series")
    train_ds = KneeStudyDataset(train_df, train_series, series_root, has_labels=True)
    val_ds = KneeStudyDataset(val_df, train_series, series_root, has_labels=True)

    model = KneeMILModel().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()

    history = []
    best_auc, best_epoch = -1.0, -1
    for epoch in range(EPOCHS):
        model.train()
        epoch_losses = []
        for i in range(len(train_ds)):
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
    ckpt_path = os.path.join(OUT_DIR, "model_v3.pt")
    torch.save({"model": model.state_dict(), "img_size": IMG_SIZE,
                "max_slices": MAX_SLICES, "k_series": K_SERIES}, ckpt_path)
    print(f"[done] saved {ckpt_path}", flush=True)
    print(f"[done] best val_macro_auc={best_auc:.4f} at epoch {best_epoch}", flush=True)

    final_val_auc, final_n, final_per_label = evaluate(model, val_ds, device)
    with open(os.path.join(OUT_DIR, "history_v3.json"), "w") as f:
        json.dump({"history": history, "best_val_macro_auc": best_auc,
                    "best_epoch": best_epoch, "final_per_label_auc": final_per_label,
                    "n_train": len(train_df), "n_val": len(val_df),
                    "torch_version": torch.__version__, "device": device}, f, indent=2)
    print(f"[done] final per-label val AUC: {final_per_label}", flush=True)

    # ---- sanity inference over the example test studies -> submission.csv
    test_csv = os.path.join(input_dir, "test.csv")
    test_series_csv = os.path.join(input_dir, "test_series.csv")
    if os.path.exists(test_csv):
        test = pd.read_csv(test_csv)
        test_series = pd.read_csv(test_series_csv)
        test_ds = KneeStudyDataset(test, test_series,
                                    os.path.join(input_dir, "test_series"), has_labels=False)
        model.eval()
        rows = []
        with torch.no_grad():
            for i in range(len(test_ds)):
                series_list, uid = test_ds[i]
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
