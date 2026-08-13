"""Plot val_macro_auc-vs-epoch for several history_v*.json runs on one chart,
to compare training recipes directly (used for the v5/v6/v7 comparison).

Usage:
    python src/plot_comparison.py out.png label1=path1.json label2=path2.json ...
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    out_path = sys.argv[1]
    runs = []
    for arg in sys.argv[2:]:
        label, path = arg.split("=", 1)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        runs.append((label, data))

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, data in runs:
        epochs = [h["epoch"] for h in data["history"]]
        aucs = [h["val_macro_auc"] for h in data["history"]]
        best = data["best_val_macro_auc"]
        ax.plot(epochs, aucs, marker="o", label=f"{label} (best={best:.3f})")

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="random (0.5)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("val macro AUC")
    ax.set_title("val macro AUC by training recipe")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
