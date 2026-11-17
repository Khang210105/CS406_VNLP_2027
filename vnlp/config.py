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
import json
import os
import subprocess
from datetime import datetime
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
    global SPLIT_PROTOCOL, TEST_VIEWS, SPLIT_NAME, DET_SPLIT_TAG, DET_TRAIN_VIEWS, DET_TRAIN_QUOTA
    global CFG, SEED, EXPERIMENT, RAW_DIR, CROPS_DIR, PROCESSED_DIR, DATA_DIR, OUTPUT_ROOT, OUTPUT_DIR
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
    EXPERIMENT = str(cfg.get("experiment") or "default")
    paths = cfg["paths"]
    RAW_DIR = _resolve(paths["raw_dir"])
    CROPS_DIR = _resolve(paths["crops_dir"])
    PROCESSED_DIR = _resolve(paths["processed_dir"])
    DATA_DIR = _resolve(paths["data_dir"])
    OUTPUT_ROOT = _resolve(paths["output_dir"])
    OUTPUT_DIR = OUTPUT_ROOT / EXPERIMENT  # mỗi experiment một thư mục kết quả riêng

    # Bộ split theo protocol: stratified -> splits/, lodo -> splits_lodo-<test_views>/
    split_cfg = cfg.get("split", {})
    SPLIT_PROTOCOL = split_cfg.get("protocol", "stratified")
    TEST_VIEWS = list(split_cfg.get("test_views") or [])
    if SPLIT_PROTOCOL == "lodo":
        if not TEST_VIEWS:
            raise ValueError("split.protocol = lodo nhưng split.test_views rỗng")
        tag = "_lodo-" + "-".join(TEST_VIEWS)
    elif SPLIT_PROTOCOL == "stratified":
        tag = ""
    else:
        raise ValueError(f"split.protocol không hợp lệ: {SPLIT_PROTOCOL} (stratified | lodo)")
    SPLIT_NAME = "splits" + tag

    # Detection có thể giới hạn view/số ảnh train (thí nghiệm RQ1) -> thư mục split riêng
    #   splits_lodo-<test_views>_tr-<view>[x<quota>]-.../detection/ ; OCR không bị ảnh hưởng
    DET_TRAIN_VIEWS = list(split_cfg.get("train_views") or [])
    DET_TRAIN_QUOTA = {str(k): int(v) for k, v in (split_cfg.get("train_quota") or {}).items()}
    det_tag = ""
    if DET_TRAIN_VIEWS or DET_TRAIN_QUOTA:
        if SPLIT_PROTOCOL != "lodo":
            raise ValueError("split.train_views / split.train_quota chỉ dùng với split.protocol = lodo")
        bad = [v for v in DET_TRAIN_QUOTA if DET_TRAIN_VIEWS and v not in DET_TRAIN_VIEWS]
        if bad:
            raise ValueError(f"split.train_quota có view không nằm trong split.train_views: {bad}")
        linked = split_cfg.get("linked_views") or {}
        clash = [v for v in set(DET_TRAIN_VIEWS) | set(DET_TRAIN_QUOTA)
                 if v in TEST_VIEWS or linked.get(v) in TEST_VIEWS]
        if clash:
            raise ValueError(f"split.train_views/train_quota chứa view thuộc tập test {TEST_VIEWS}: {clash} "
                             "-> tập train detection sẽ rỗng")
        train_views = DET_TRAIN_VIEWS or sorted(DET_TRAIN_QUOTA)
        det_tag = "_tr-" + "-".join(f"{v}x{DET_TRAIN_QUOTA[v]}" if v in DET_TRAIN_QUOTA else v
                                    for v in train_views)

    # Recognition (OCR)
    RECOG_SPLIT_DIR = DATA_DIR / SPLIT_NAME / "recognition"
    RECOG_TRAIN_DIR = RECOG_SPLIT_DIR / "train"
    RECOG_VAL_DIR = RECOG_SPLIT_DIR / "val"
    RECOG_TEST_DIR = RECOG_SPLIT_DIR / "test"
    RECOG_MANIFEST = DATA_DIR / f"recognition_split{tag}.csv"
    RECOG_OUTPUT_DIR = OUTPUT_DIR / "recognition"
    CRNN_OUTPUT_DIR = RECOG_OUTPUT_DIR / "crnn"
    PARSEQ_OUTPUT_DIR = RECOG_OUTPUT_DIR / "parseq"
    CRNN_WEIGHTS = CRNN_OUTPUT_DIR / "ocr_model.pth"
    PARSEQ_WEIGHTS = PARSEQ_OUTPUT_DIR / "parseq_model.ckpt"

    # Detection (YOLO)
    DET_SPLIT_TAG = det_tag
    DET_SPLIT_DIR = DATA_DIR / (SPLIT_NAME + det_tag) / "detection"
    DET_DATA_YAML = DET_SPLIT_DIR / "data.yaml"
    DET_MANIFEST = DATA_DIR / f"detection_split{tag}{det_tag}.csv"
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


def _git_commit():
    """Commit hiện tại; trên Kaggle (không có .git) đọc BUILD_INFO.json do kaggle_push ghi."""
    build_info = REPO_ROOT / "BUILD_INFO.json"
    if build_info.is_file():
        return json.loads(build_info.read_text(encoding="utf-8")).get("git_commit")
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                                capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return f"{commit}{'+dirty' if dirty else ''}" if commit else None
    except Exception:
        return None


def snapshot(out_dir=None):
    """Ghi config_used.yaml (config thực tế sau ghi đè) + run_info.json vào out_dir.
    Gọi ở đầu mỗi lần train -> biết kết quả nào ứng với code/tham số nào."""
    out_dir = Path(out_dir or OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(CFG, f, allow_unicode=True, sort_keys=False)
    info = {
        "experiment": EXPERIMENT,
        "time": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "device": get_device(),
        "on_kaggle": Path("/kaggle/input").exists(),
    }
    (out_dir / "run_info.json").write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding="utf-8")
    return info


def pick(value, *keys):
    """Trả về value nếu khác None, ngược lại lấy CFG[keys...]. Dùng cho tham số mặc định của hàm."""
    if value is not None:
        return value
    node = CFG
    for k in keys:
        node = node[k]
    return node


reload()
