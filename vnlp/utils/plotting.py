"""
plotting.py
===========
Biểu đồ dùng chung: đường cong học (history) và metric theo view.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_history(history, out_path, title, acc_keys=("val_acc",)):
    df = pd.DataFrame(history)
    if df.empty:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for key, color in (("train_loss", "#8884d8"), ("val_loss", "#82ca9d")):
        if key in df and not df[key].isna().all():
            ax1.plot(df["epoch"], df[key], label=key, color=color, linewidth=2)
    ax1.set_title(f"{title} - Loss", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    for key in acc_keys:
        if key in df:
            ax2.plot(df["epoch"], df[key], label=key, linewidth=2)
    ax2.set_title(f"{title} - Accuracy (%)", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_metrics_by_view(df_metrics, out_path, title):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = range(len(df_metrics))
    ax.bar([i - 0.2 for i in x], df_metrics["plate_acc"], width=0.4, label="Plate Acc (%)", color="#2ca02c")
    ax.bar([i + 0.2 for i in x], df_metrics["cer"], width=0.4, label="CER (%)", color="#d62728")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{v}\n(n={n})" for v, n in zip(df_metrics["view"], df_metrics["samples"])])
    ax.set_ylim(0, 110)
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
