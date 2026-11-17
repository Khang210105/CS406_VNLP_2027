"""
evaluate.py
===========
Đánh giá YOLO phát hiện biển số trên tập TEST: Precision, Recall, mAP50, mAP50-95.

    python -m vnlp.detection.evaluate

    from vnlp.detection.evaluate import evaluate
    df = evaluate()
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from ultralytics import YOLO

from vnlp import config


def evaluate(weights=None, data_yaml=None, out_dir=None, **overrides):
    """kwargs ghi đè config.yaml -> detection.eval (imgsz, batch, ...)."""
    weights = Path(weights or config.YOLO_WEIGHTS)
    data_yaml = data_yaml or config.DET_DATA_YAML
    out_dir = out_dir or config.DET_OUTPUT_DIR
    if not weights.is_file():
        raise FileNotFoundError(f"Không tìm thấy trọng số YOLO: {weights}. Hãy train trước.")

    params = {**config.CFG["detection"]["eval"], **overrides}
    params.setdefault("device", 0 if torch.cuda.is_available() else "cpu")
    print("🚀 ĐANG ĐÁNH GIÁ TRÊN TẬP TEST...")
    metrics = YOLO(str(weights)).val(
        data=str(data_yaml),
        split="test",
        **params,
        project=str(out_dir),
        name="test_eval",
        exist_ok=True,
        verbose=True,
    )

    r = metrics.results_dict
    df = pd.DataFrame({
        "Metric": ["Precision", "Recall", "mAP50", "mAP50-95"],
        "Value_Percent": [round(100 * r[k], 2) for k in (
            "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)")],
    })
    print("\n📊 KẾT QUẢ DETECTION (TEST):")
    print(df.to_string(index=False))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "yolo_test_metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(df["Metric"], df["Value_Percent"], color=["#1f77b4", "#aec7e8", "#2ca02c", "#98df8a"],
                  width=0.5, edgecolor="black")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"{bar.get_height()}%",
                ha="center", va="bottom", fontweight="bold")
    ax.set_title("YOLO Bounding Box Evaluation Metrics (Test)", fontweight="bold")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_dir / "yolo_test_metrics.png", dpi=200)
    plt.close(fig)
    print(f"💾 Đã lưu vào: {out_dir}")
    return df


if __name__ == "__main__":
    evaluate()
