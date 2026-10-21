"""
plot_training.py
================
Vẽ biểu đồ độ chính xác & loss theo epoch từ file results.csv
mà YOLO tự sinh ra sau khi train.

Cách dùng:
    python plot_training.py
    python plot_training.py --csv runs/detect/bien_so_detector/results.csv
    python plot_training.py --csv results.csv --save bieu_do.png
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

# =====================================================
# CONFIG MẶC ĐỊNH
# =====================================================

DEFAULT_CSV  = "runs/detect/bien_so_detector-2/results.csv"
DEFAULT_SAVE = "training_results_2.png"

# =====================================================
# CÁC CỘT CẦN VẼ  (tên cột YOLO có khoảng trắng thừa → strip)
# metrics/precision(B) : Precision trên tập val
# metrics/recall(B)    : Recall
# metrics/mAP50(B)     : mAP@0.5
# metrics/mAP50-95(B)  : mAP@0.5:0.95
# train/box_loss       : Box regression loss
# train/cls_loss       : Classification loss
# val/box_loss         : Val box loss
# val/cls_loss         : Val cls loss
# =====================================================

METRIC_PANELS = [
    {
        "title": "Precision & Recall",
        "series": [
            ("metrics/precision(B)", "Precision", "#00d4aa"),
            ("metrics/recall(B)",    "Recall",    "#ff6b6b"),
        ],
        "ylabel": "Score",
        "ylim": (0, 1.05),
    },
    {
        "title": "mAP",
        "series": [
            ("metrics/mAP50(B)",    "mAP@0.5",      "#4ecdc4"),
            ("metrics/mAP50-95(B)", "mAP@0.5:0.95", "#ffe66d"),
        ],
        "ylabel": "mAP",
        "ylim": (0, 1.05),
    },
    {
        "title": "Train Loss",
        "series": [
            ("train/box_loss", "Box loss", "#a78bfa"),
            ("train/cls_loss", "Cls loss", "#fb923c"),
        ],
        "ylabel": "Loss",
        "ylim": None,
    },
    {
        "title": "Validation Loss",
        "series": [
            ("val/box_loss", "Val box loss", "#60a5fa"),
            ("val/cls_loss", "Val cls loss", "#f472b6"),
        ],
        "ylabel": "Loss",
        "ylim": None,
    },
]


# =====================================================
# HELPERS
# =====================================================

def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Không tìm thấy file: {path}\n"
            "Hãy chạy train_yolo.py trước, hoặc chỉ định đúng đường dẫn "
            "bằng --csv"
        )
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()   # YOLO hay thêm khoảng trắng
    return df


def best_epoch_info(df: pd.DataFrame) -> str:
    col = "metrics/mAP50(B)"
    if col not in df.columns:
        return ""
    best_idx  = df[col].idxmax()
    best_ep   = int(df.loc[best_idx, "epoch"]) + 1   # 0-indexed → 1-indexed
    best_map  = df.loc[best_idx, col]
    prec_col  = "metrics/precision(B)"
    rec_col   = "metrics/recall(B)"
    prec = df.loc[best_idx, prec_col] if prec_col in df.columns else float("nan")
    rec  = df.loc[best_idx, rec_col]  if rec_col  in df.columns else float("nan")
    return (
        f"Best epoch: {best_ep}  |  "
        f"mAP@0.5 = {best_map:.4f}  |  "
        f"Precision = {prec:.4f}  |  "
        f"Recall = {rec:.4f}"
    )


# =====================================================
# PLOT
# =====================================================

def plot(df: pd.DataFrame, save_path: str):

    epochs = df["epoch"] + 1   # 0-indexed → 1-indexed

    # Dark theme
    plt.style.use("dark_background")

    fig = plt.figure(figsize=(16, 10), facecolor="#0d1117")
    fig.suptitle(
        "YOLO Training Dashboard — Biển số xe",
        fontsize=18, fontweight="bold", color="#e6edf3",
        y=0.97
    )

    gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]

    for ax, panel in zip(axes, METRIC_PANELS):

        ax.set_facecolor("#161b22")
        ax.set_title(panel["title"], color="#e6edf3", fontsize=12, pad=8)
        ax.set_xlabel("Epoch", color="#8b949e", fontsize=9)
        ax.set_ylabel(panel["ylabel"], color="#8b949e", fontsize=9)
        ax.tick_params(colors="#8b949e", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        ax.grid(color="#21262d", linewidth=0.8, linestyle="--")

        if panel["ylim"]:
            ax.set_ylim(*panel["ylim"])

        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        best_ep_line_drawn = False

        for col, label, color in panel["series"]:
            if col not in df.columns:
                print(f"[WARN] Cột '{col}' không có trong CSV, bỏ qua.")
                continue

            values = df[col]
            ax.plot(epochs, values, label=label, color=color,
                    linewidth=1.8, alpha=0.9)

            # Đánh dấu giá trị tốt nhất
            if "loss" in col.lower():
                best_i = values.idxmin()
            else:
                best_i = values.idxmax()

            bx, by = epochs.iloc[best_i], values.iloc[best_i]
            ax.scatter(bx, by, color=color, s=60, zorder=5)
            ax.annotate(
                f"{by:.3f}",
                xy=(bx, by),
                xytext=(6, 6), textcoords="offset points",
                color=color, fontsize=7.5
            )

            # Đường dọc epoch tốt nhất (chỉ vẽ 1 lần cho panel)
            if not best_ep_line_drawn and "loss" not in col.lower():
                map_col = "metrics/mAP50(B)"
                if map_col in df.columns:
                    global_best = df[map_col].idxmax()
                    ax.axvline(
                        x=epochs.iloc[global_best],
                        color="#ffe66d", linewidth=0.8,
                        linestyle=":", alpha=0.6, label=f"Best epoch"
                    )
                    best_ep_line_drawn = True

        ax.legend(fontsize=8, framealpha=0.3, labelcolor="white")

    # Footer: thông tin epoch tốt nhất
    info = best_epoch_info(df)
    if info:
        fig.text(
            0.5, 0.01, info,
            ha="center", fontsize=9.5,
            color="#58a6ff",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#161b22",
                      edgecolor="#30363d", alpha=0.9)
        )

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"✅ Biểu đồ đã lưu: {save_path}")
    plt.show()


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Vẽ biểu đồ training YOLO")
    parser.add_argument(
        "--csv",  default=DEFAULT_CSV,
        help=f"Đường dẫn tới results.csv (mặc định: {DEFAULT_CSV})"
    )
    parser.add_argument(
        "--save", default=DEFAULT_SAVE,
        help=f"Tên file ảnh output (mặc định: {DEFAULT_SAVE})"
    )
    args = parser.parse_args()

    df = load_csv(args.csv)
    print(f"Đọc được {len(df)} epochs từ: {args.csv}")
    print(best_epoch_info(df))
    plot(df, args.save)