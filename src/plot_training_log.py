"""Parse a downloaded Kaggle kernel log (`kaggle kernels output ... `) and plot
the per-epoch training loss curve.

Usage:
    python src/plot_training_log.py path/to/kernel.log out.png
"""
import json
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPOCH_RE = re.compile(r"\[epoch (\d+)/(\d+)\] mean_loss=([\d.]+)")


def parse_log(log_path):
    epochs, losses = [], []
    with open(log_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Kaggle log files are a stream of JSON objects (one per printed line),
    # not one JSON document -- pull out every "data" field with a regex match
    # rather than trying to json.loads() the whole file.
    for line in text.splitlines():
        m = re.search(r'"data"\s*:\s*"((?:[^"\\]|\\.)*)"', line)
        if not m:
            continue
        msg = json.loads(f'"{m.group(1)}"')  # unescape \n etc.
        em = EPOCH_RE.search(msg)
        if em:
            epochs.append(int(em.group(1)))
            losses.append(float(em.group(3)))
    return epochs, losses


def main():
    log_path, out_path = sys.argv[1], sys.argv[2]
    epochs, losses = parse_log(log_path)
    if not epochs:
        print("no epoch lines found in log")
        return

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, losses, marker="o")
    plt.xlabel("epoch")
    plt.ylabel("mean training loss (BCE)")
    plt.title("Baseline (v1) training loss -- 58 labeled studies")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"saved {out_path} ({len(epochs)} epochs, "
          f"loss {losses[0]:.4f} -> {losses[-1]:.4f})")


if __name__ == "__main__":
    main()
