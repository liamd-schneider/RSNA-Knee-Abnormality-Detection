"""Read one DICOM series (a folder of .dcm slices) into a normalized volume.

A "series" is one MRI acquisition (e.g. sagittal T2) -- a stack of 2D slices.
A "study" is a whole knee exam and usually has ~5 series (different planes/
sequences). This module only deals with a single series; stacking series into
a study-level input happens in dataset.py.
"""
import os

import numpy as np
import pydicom

IMG_SIZE = 128  # small on purpose for fast local CPU iteration; raise for real training
MAX_SLICES = 32


def _resize(arr, size):
    from PIL import Image
    img = Image.fromarray(arr.astype(np.float32))
    return np.asarray(img.resize((size, size), Image.BILINEAR), dtype=np.float32)


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
    """Return a (n_slices, img_size, img_size) float32 array in [0, 1], or None
    if the series has no readable slices."""
    files = [os.path.join(series_dir, f) for f in os.listdir(series_dir)
             if f.endswith(".dcm")]
    if not files:
        return None

    slices = []
    for f in files:
        try:
            ds = pydicom.dcmread(f)
            arr = ds.pixel_array.astype(np.float32)
        except Exception:
            # A handful of DICOMs in this corpus have truncated/corrupted
            # pixel data (pydicom raises ValueError on the byte-count
            # mismatch) -- one bad slice out of thousands shouldn't crash a
            # run hours into training, so it's dropped and the rest of the
            # series is used instead.
            continue

        # DICOM stores raw sensor values; RescaleSlope/Intercept convert them
        # to the actual physical intensity the scanner measured.
        slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
        intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
        arr = arr * slope + intercept

        # MONOCHROME1 means "0 = white" instead of the usual "0 = black" --
        # some scanners emit this, and skipping the check would silently feed
        # visually-inverted images into the model.
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            arr = arr.max() - arr

        # File names are opaque UIDs, not slice order -- InstanceNumber is the
        # field that actually encodes where a slice sits in the stack.
        inst = getattr(ds, "InstanceNumber", None)
        slices.append((int(inst) if inst is not None else 0, arr))

    if not slices:
        return None

    slices.sort(key=lambda x: x[0])
    vol = np.stack([a for _, a in slices])

    # Windowing to [0, 1] off the 1st/99th percentile of *this* series, not a
    # fixed global range -- intensity scales vary a lot between scanners/
    # protocols, and a fixed window would let one unusually bright series
    # wreck the contrast for everything trained alongside it.
    lo, hi = np.percentile(vol, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    vol = np.clip((vol - lo) / (hi - lo), 0, 1)

    if vol.shape[0] > max_slices:
        start = (vol.shape[0] - max_slices) // 2
        vol = vol[start:start + max_slices]

    vol = np.stack([_resize(_pad_to_square(s), img_size) for s in vol])
    return vol
