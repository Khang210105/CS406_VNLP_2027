"""
io.py
=====
Đọc ảnh / mẫu OCR.

Format thư mục split OCR:  <split_dir>/<sample_id>/image.jpg + label.txt
sample_id có dạng "view4_00012" -> view_of() lấy được view để báo cáo theo từng view.
"""

from pathlib import Path

import cv2
import numpy as np


def imread(path, flags=cv2.IMREAD_COLOR):
    """cv2.imread trên Windows trả None với đường dẫn có dấu (vd 'Đồ án') -> đọc bytes rồi decode."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(data, flags) if data.size else None


def normalize_label(text: str) -> str:
    return text.strip().replace("-", "").replace(".", "").replace("\n", "").replace(" ", "")


def view_of(sample_id: str) -> str:
    return sample_id.split("_")[0]


def load_samples(root_dir):
    """Trả về list (image_path, label, sample_id) đã sắp xếp theo sample_id."""
    root_dir = Path(root_dir)
    if not root_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục split: {root_dir}")

    samples = []
    for sample_dir in sorted(p for p in root_dir.iterdir() if p.is_dir()):
        image_path = sample_dir / "image.jpg"
        label_path = sample_dir / "label.txt"
        if not image_path.is_file() or not label_path.is_file():
            continue
        text = normalize_label(label_path.read_text(encoding="utf-8-sig"))
        if text:
            samples.append((str(image_path), text, sample_dir.name))
    return samples
