"""
config.py
=========
Đọc config.yaml ở gốc repo và tính sẵn các đường dẫn hay dùng.

    from vnlp import config
    config.CFG["crnn"]["epochs"]      # giá trị thô từ YAML
    config.RECOG_TRAIN_DIR            # đường dẫn đã tính sẵn (Path tuyệt đối)
    config.reload({"crnn.epochs": 5}) # ghi đè tạm thời (vd trong notebook)

Ưu tiên: reload(overrides) > biến môi trường > config.yaml.
Các hàm train/eval đọc config.* lúc CHẠY (không phải lúc import),
nên reload() có hiệu lực ngay với lần gọi tiếp theo.
"""

import copy
import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(os.getenv("VNLP_CONFIG", REPO_ROOT / "config.yaml"))

with open(CONFIG_PATH, encoding="utf-8") as _f:
    _BASE = yaml.safe_load(_f)

CFG: dict = {}


def _resolve(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def _set_dotted(d, key, value):
    *parents, last = key.split(".")
    for k in parents:
        d = d.setdefault(k, {})
    d[last] = value


def reload(overrides: dict | None = None):
    """Nạp lại config.yaml (+ env + overrides dạng {"a.b.c": value}) và tính lại mọi đường dẫn."""
    global CFG, SEED, RAW_DIR, CROPS_DIR, PROCESSED_DIR, DATA_DIR, OUTPUT_DIR
    global RECOG_SPLIT_DIR, RECOG_TRAIN_DIR, RECOG_VAL_DIR, RECOG_TEST_DIR, RECOG_MANIFEST
    global RECOG_OUTPUT_DIR, CRNN_OUTPUT_DIR, PARSEQ_OUTPUT_DIR, CRNN_WEIGHTS, PARSEQ_WEIGHTS
    global DET_SPLIT_DIR, DET_DATA_YAML, DET_MANIFEST, DET_OUTPUT_DIR, DET_RUN_NAME, YOLO_WEIGHTS
    global CHARS, IMG_W, IMG_H, MAX_LABEL_LENGTH

    cfg = copy.deepcopy(_BASE)
    if os.getenv("VNLP_DATA_DIR"):
        cfg["paths"]["data_dir"] = os.environ["VNLP_DATA_DIR"]
    if os.getenv("VNLP_OUTPUT_DIR"):
        cfg["paths"]["output_dir"] = os.environ["VNLP_OUTPUT_DIR"]
    for key, value in (overrides or {}).items():
        _set_dotted(cfg, key, value)
    CFG = cfg

    SEED = cfg["seed"]
    paths = cfg["paths"]
    RAW_DIR = _resolve(paths["raw_dir"])
    CROPS_DIR = _resolve(paths["crops_dir"])
    PROCESSED_DIR = _resolve(paths["processed_dir"])
    DATA_DIR = _resolve(paths["data_dir"])
    OUTPUT_DIR = _resolve(paths["output_dir"])

    # Recognition (OCR)
    RECOG_SPLIT_DIR = DATA_DIR / "splits" / "recognition"
    RECOG_TRAIN_DIR = RECOG_SPLIT_DIR / "train"
    RECOG_VAL_DIR = RECOG_SPLIT_DIR / "val"
    RECOG_TEST_DIR = RECOG_SPLIT_DIR / "test"
    RECOG_MANIFEST = DATA_DIR / "recognition_split.csv"
    RECOG_OUTPUT_DIR = OUTPUT_DIR / "recognition"
    CRNN_OUTPUT_DIR = RECOG_OUTPUT_DIR / "crnn"
    PARSEQ_OUTPUT_DIR = RECOG_OUTPUT_DIR / "parseq"
    CRNN_WEIGHTS = CRNN_OUTPUT_DIR / "ocr_model.pth"
    PARSEQ_WEIGHTS = PARSEQ_OUTPUT_DIR / "parseq_model.ckpt"

    # Detection (YOLO)
    DET_SPLIT_DIR = DATA_DIR / "splits" / "detection"
    DET_DATA_YAML = DET_SPLIT_DIR / "data.yaml"
    DET_MANIFEST = DATA_DIR / "detection_split.csv"
    DET_OUTPUT_DIR = OUTPUT_DIR / "detection"
    DET_RUN_NAME = cfg["detection"]["run_name"]
    YOLO_WEIGHTS = DET_OUTPUT_DIR / DET_RUN_NAME / "weights" / "best.pt"

    # OCR chung
    CHARS = cfg["ocr"]["charset"]
    IMG_W = cfg["ocr"]["img_w"]
    IMG_H = cfg["ocr"]["img_h"]
    MAX_LABEL_LENGTH = cfg["ocr"]["max_label_length"]
    return CFG


def get_device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def default_num_workers() -> int:
    value = CFG.get("num_workers", "auto")
    if value == "auto":
        return 0 if os.name == "nt" else 4  # Windows + nhiều worker hay lỗi/chậm
    return int(value)


def pick(value, *keys):
    """Trả về value nếu khác None, ngược lại lấy CFG[keys...]. Dùng cho tham số mặc định của hàm."""
    if value is not None:
        return value
    node = CFG
    for k in keys:
        node = node[k]
    return node


reload()
