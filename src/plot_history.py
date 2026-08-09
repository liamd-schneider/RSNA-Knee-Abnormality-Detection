"""Plot train_loss + val_macro_auc from a history_v*.json file (written by
kernels/train_v2, train_v3, ...).

Usage:
    python src/plot_history.py path/to/history.json out.png
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    hist_path, out_path = sys.argv[1], sys.argv[2]
    with open(hist_path, encoding="utf-8") as f:
        data = json.load(f)

    epochs = [h["epoch"] for h in data["history"]]
    train_loss = [h["train_loss"] for h in data["history"]]
    val_auc = [h["val_macro_auc"] for h in data["history"]]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(epochs, train_loss, marker="o", color="tab:blue", label="train loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train loss (BCE)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(epochs, val_auc, marker="s", color="tab:orange", label="val macro AUC")
    ax2.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="random (0.5)")
    ax2.set_ylabel("val macro AUC", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    n_train, n_val = data.get("n_train"), data.get("n_val")
    plt.title(f"train loss vs. held-out val AUC (n_train={n_train}, n_val={n_val})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"saved {out_path}")
    print(f"best_val_macro_auc={data['best_val_macro_auc']:.4f} at epoch {data['best_epoch']}")
    print(f"final_per_label_auc={data['final_per_label_auc']}")


if __name__ == "__main__":
    main()
