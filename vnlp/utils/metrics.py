"""
metrics.py
==========
Metric OCR: Plate Accuracy (đúng toàn biển) và CER (edit distance / số ký tự nhãn),
tổng và theo từng view.
"""

from pathlib import Path

import pandas as pd

from vnlp.utils.io import view_of
from vnlp.utils.plotting import plot_metrics_by_view


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def summarize_predictions(records):
    """
    records: list dict {sample_id, gt, pred}
    Trả về (df_predictions, df_metrics). df_metrics có 1 dòng ALL + 1 dòng mỗi view.
    """
    df = pd.DataFrame(records)
    df["view"] = df["sample_id"].map(view_of)
    df["correct"] = df["gt"] == df["pred"]
    df["char_errors"] = [edit_distance(g, p) for g, p in zip(df["gt"], df["pred"])]
    df["gt_len"] = df["gt"].str.len()

    def row(name, part):
        return {
            "view": name,
            "samples": len(part),
            "plate_acc": round(100 * part["correct"].mean(), 2),
            "cer": round(100 * part["char_errors"].sum() / max(part["gt_len"].sum(), 1), 2),
        }

    rows = [row("ALL", df)] + [row(v, g) for v, g in sorted(df.groupby("view"))]
    return df, pd.DataFrame(rows)


def save_eval(df_pred, df_metrics, out_dir, prefix):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df_pred.to_csv(out_dir / f"{prefix}_test_predictions.csv", index=False, encoding="utf-8")
    df_metrics.to_csv(out_dir / f"{prefix}_test_metrics.csv", index=False, encoding="utf-8")
    plot_metrics_by_view(df_metrics, out_dir / f"{prefix}_test_metrics.png",
                         f"{prefix.upper()} - kết quả trên tập TEST theo view")


def print_eval(df_pred, df_metrics, model_name, max_errors=50):
    wrong = df_pred[~df_pred["correct"]]
    print(f"\n🚨 {model_name}: {len(wrong)}/{len(df_pred)} biển đọc sai"
          + (f" (in {max_errors} mẫu đầu)" if len(wrong) > max_errors else ""))
    for _, r in wrong.head(max_errors).iterrows():
        print(f"  {r['sample_id']:<14} GT: {r['gt']:<12} Pred: {r['pred']:<12} ({r['char_errors']} lỗi)")
    print(f"\n📊 {model_name} trên tập TEST:")
    print(df_metrics.to_string(index=False))
