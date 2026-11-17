"""
crop_plates.py
==============
Bước 1 của quy trình data: cắt biển số từ ảnh gốc theo bounding box YOLO.

  raw_dir/<view>/{images,labels}  ->  crops_dir/<view>/<tên ảnh>[__boxNNN]/<ảnh crop> + label.txt (rỗng)

Ảnh có 1 biển: folder = tên ảnh gốc. Nhiều biển: thêm hậu tố __box001, __box002...
Tham số: config.yaml -> paths.raw_dir, paths.crops_dir, crop.views, crop.margin

    python -m vnlp.data.crop_plates
Tiếp theo: python -m vnlp.data.label_plates (gợi ý nhãn) -> review tay -> processed_dir.
"""

import math
from pathlib import Path

from PIL import Image

from vnlp import config

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def crop_view(view_name, view_dir, output_root, margin=0.0):
    """Crop YOLO boxes; save each crop and an empty label.txt in its own folder."""
    image_dir = view_dir / "images"
    label_dir = view_dir / "labels"
    view_output_dir = output_root / view_name
    view_output_dir.mkdir(parents=True, exist_ok=True)

    if not image_dir.is_dir():
        print(f"[{view_name}] Bỏ qua: không tìm thấy {image_dir}")
        return {"images": 0, "crops": 0, "missing_labels": 0, "invalid_boxes": 0}

    image_paths = [p for p in image_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    total_crops, missing_labels, invalid_boxes = 0, 0, 0

    for image_path in image_paths:
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            missing_labels += 1
            print(f"[{view_name}] Thiếu label: {image_path.name}")
            continue

        try:
            with Image.open(image_path) as im:
                image = im.convert("RGB")
        except Exception as exc:
            print(f"[{view_name}] Không đọc được ảnh {image_path.name}: {exc}")
            continue

        img_w, img_h = image.size
        try:
            lines = label_path.read_text(encoding="utf-8").splitlines()
        except Exception as exc:
            print(f"[{view_name}] Không đọc được label {label_path.name}: {exc}")
            continue

        boxes = []
        for line_no, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                invalid_boxes += 1
                print(f"[{view_name}] Nhãn sai định dạng: {label_path.name}, dòng {line_no}")
                continue
            try:
                class_id = int(parts[0])
                x_center, y_center, box_w, box_h = map(float, parts[1:])
            except ValueError:
                invalid_boxes += 1
                print(f"[{view_name}] Nhãn không hợp lệ: {label_path.name}, dòng {line_no}")
                continue

            values = (x_center, y_center, box_w, box_h)
            if (class_id < 0 or not all(math.isfinite(v) for v in values)
                    or not (0 <= x_center <= 1 and 0 <= y_center <= 1)
                    or not (0 < box_w <= 1 and 0 < box_h <= 1)):
                invalid_boxes += 1
                print(f"[{view_name}] Tọa độ không hợp lệ: {label_path.name}, dòng {line_no}")
                continue

            x1 = (x_center - box_w / 2) * img_w
            y1 = (y_center - box_h / 2) * img_h
            x2 = (x_center + box_w / 2) * img_w
            y2 = (y_center + box_h / 2) * img_h
            pad_x, pad_y = (x2 - x1) * margin, (y2 - y1) * margin
            left = max(0, int(math.floor(x1 - pad_x)))
            top = max(0, int(math.floor(y1 - pad_y)))
            right = min(img_w, int(math.ceil(x2 + pad_x)))
            bottom = min(img_h, int(math.ceil(y2 + pad_y)))

            if right <= left or bottom <= top:
                invalid_boxes += 1
                print(f"[{view_name}] Box rỗng: {label_path.name}, dòng {line_no}")
                continue

            boxes.append((left, top, right, bottom))

        for box_index, (left, top, right, bottom) in enumerate(boxes, start=1):
            # If there is one plate, folder name is the source image stem.
            # If there are multiple plates, append an index to avoid collisions.
            folder_name = (image_path.stem if len(boxes) == 1
                           else f"{image_path.stem}__box{box_index:03d}")
            crop_dir = view_output_dir / folder_name
            crop_dir.mkdir(parents=True, exist_ok=True)

            crop = image.crop((left, top, right, bottom))
            # Keep the original filename, even though the image content is cropped.
            crop.save(crop_dir / image_path.name)
            (crop_dir / "label.txt").touch(exist_ok=True)
            total_crops += 1

    summary = {
        "images": len(image_paths),
        "crops": total_crops,
        "missing_labels": missing_labels,
        "invalid_boxes": invalid_boxes,
    }
    print(
        f"[{view_name}] Ảnh nguồn: {summary['images']} | Crop: {total_crops} | "
        f"Thiếu label: {missing_labels} | Box lỗi: {invalid_boxes}\n"
        f"    Lưu tại: {view_output_dir}"
    )
    return summary


def main():
    cfg = config.CFG["crop"]
    margin = float(cfg.get("margin", 0.0))
    if not 0 <= margin <= 1:
        raise ValueError("crop.margin phải nằm trong khoảng 0 đến 1")
    if not config.RAW_DIR.is_dir():
        raise FileNotFoundError(f"Không tìm thấy raw_dir: {config.RAW_DIR}")

    views = cfg.get("views") or sorted(p.name for p in config.RAW_DIR.iterdir() if p.is_dir())
    summaries = [crop_view(v, config.RAW_DIR / v, config.CROPS_DIR, margin) for v in views]
    print()
    print("=== TỔNG KẾT ===")
    print(f"Số view: {len(summaries)} | Tổng crop: {sum(s['crops'] for s in summaries)} | "
          f"Thiếu label: {sum(s['missing_labels'] for s in summaries)} | "
          f"Box lỗi: {sum(s['invalid_boxes'] for s in summaries)}")


if __name__ == "__main__":
    main()
