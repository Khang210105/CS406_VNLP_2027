from pathlib import Path
import argparse
import math
from PIL import Image

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
    parser = argparse.ArgumentParser(description="Crop biển số từ ảnh theo bounding box YOLO.")
    parser.add_argument("--raw-root", type=Path, default=Path("dataset/raw"), help="Thư mục chứa các view.")
    parser.add_argument("--processed-root", type=Path, default=Path("dataset/processed"), help="Thư mục lưu crop.")
    parser.add_argument("--views", nargs="*", default=None, help="Tên view cần chạy; mặc định chạy tất cả thư mục con trong raw.")
    parser.add_argument("--margin", type=float, default=0.0, help="Nới box theo tỷ lệ mỗi phía, ví dụ 0.05.")
    args = parser.parse_args()

    if not 0 <= args.margin <= 1:
        parser.error("--margin phải nằm trong khoảng 0 đến 1.")
    if not args.raw_root.is_dir():
        parser.error(f"Không tìm thấy raw root: {args.raw_root}")

    view_dirs = [args.raw_root / name for name in args.views] if args.views else sorted(p for p in args.raw_root.iterdir() if p.is_dir())
    if not view_dirs:
        parser.error(f"Không tìm thấy view nào trong {args.raw_root}")

    summaries = [crop_view(p.name, p, args.processed_root, args.margin) for p in view_dirs]
    print("\n=== TỔNG KẾT ===")
    print(f"Số view: {len(summaries)} | Tổng crop: {sum(s['crops'] for s in summaries)} | Thiếu label: {sum(s['missing_labels'] for s in summaries)} | Box lỗi: {sum(s['invalid_boxes'] for s in summaries)}")


if __name__ == "__main__":
    main()

# python ./dataset/processed/crop_plates.py --raw-root dataset/raw --processed-root dataset/processed --views view1 view2 view3 view4