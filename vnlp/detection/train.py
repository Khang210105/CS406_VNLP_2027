"""
train.py
========
Huấn luyện YOLO phát hiện vùng biển số. Siêu tham số: config.yaml -> detection.train

    python -m vnlp.detection.train

    from vnlp.detection.train import train
    best_pt = train(epochs=50)          # kwargs ghi đè detection.train
"""

from pathlib import Path

import torch
from ultralytics import YOLO

from vnlp import config


def train(data_yaml=None, out_dir=None, name=None, base_weights=None, **overrides):
    data_yaml = Path(data_yaml or config.DET_DATA_YAML)
    out_dir = Path(out_dir or config.DET_OUTPUT_DIR)
    name = name or config.DET_RUN_NAME
    base_weights = base_weights or config.CFG["detection"]["base_weights"]
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Không tìm thấy {data_yaml}. Chạy: python -m vnlp.data.split_detection")

    params = {**config.CFG["detection"]["train"], **overrides}
    config.snapshot(out_dir / name)
    params.setdefault("device", 0 if torch.cuda.is_available() else "cpu")
    params.setdefault("workers", config.default_num_workers())
    params.setdefault("seed", config.SEED)

    print("🚀 BẮT ĐẦU HUẤN LUYỆN YOLO:", params)
    YOLO(base_weights).train(data=str(data_yaml), project=str(out_dir), name=name,
                             exist_ok=True, cache=False, **params)
    best = out_dir / name / "weights" / "best.pt"
    print(f"✅ HUẤN LUYỆN HOÀN TẤT! Model: {best}")
    return best


if __name__ == "__main__":
    train()
