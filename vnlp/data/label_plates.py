"""
label_plates.py
===============
Bước 2 của quy trình data: dùng PARSeq đã train đọc thử từng crop và ghi kết quả vào
label.txt làm NHÃN GỢI Ý. Sau đó review/sửa tay rồi lưu sang paths.processed_dir.

  crops_dir/<view>/<sample>/label.txt  <-  dự đoán PARSeq

Tham số: config.yaml -> paths.crops_dir, prelabel.views, prelabel.overwrite
Weights: config.PARSEQ_WEIGHTS (outputs/recognition/parseq/parseq_model.ckpt)

    python -m vnlp.data.label_plates
"""

import cv2
import torch

from vnlp import config
from vnlp.recognition import parseq
from vnlp.recognition.preprocess import preprocess_plate
from vnlp.utils.io import imread

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def predict_plate(model, transform, image_path, device):
    image = imread(image_path)
    if image is None:
        raise ValueError(f"Không đọc được ảnh: {image_path}")
    rgb = cv2.cvtColor(preprocess_plate(image), cv2.COLOR_GRAY2RGB)
    with torch.no_grad():
        return parseq.decode(model, transform(rgb).unsqueeze(0).to(device))[0]


def label_view(model, transform, view, device, overwrite=False):
    view_dir = config.CROPS_DIR / view
    if not view_dir.is_dir():
        print(f"[{view}] bỏ qua: không có {view_dir}")
        return 0
    folders = sorted(p for p in view_dir.iterdir() if p.is_dir())
    print(f"\n===== {view}: {len(folders)} crop =====")

    written = 0
    for folder in folders:
        label_path = folder / "label.txt"
        if not overwrite and label_path.is_file() and label_path.read_text(encoding="utf-8-sig").strip():
            continue  # đã có nhãn -> giữ nguyên
        images = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        if not images:
            print(f"Bỏ qua: {folder.name} (không có ảnh)")
            continue
        try:
            prediction = predict_plate(model, transform, images[0], device)
            label_path.write_text(prediction, encoding="utf-8")
            written += 1
            print(f"{folder.name}: {prediction}")
        except Exception as exc:  # 1 ảnh lỗi không làm dừng cả view
            print(f"Lỗi {folder.name}: {exc}")
    return written


def main():
    cfg = config.CFG["prelabel"]
    device = config.get_device()
    model = parseq.load_model(config.PARSEQ_WEIGHTS, device)
    transform = parseq.build_transform(augment=False)
    total = sum(label_view(model, transform, v, device, cfg.get("overwrite", False)) for v in cfg["views"])
    print(f"\nHoàn tất: ghi {total} nhãn gợi ý vào {config.CROPS_DIR}")


if __name__ == "__main__":
    main()
