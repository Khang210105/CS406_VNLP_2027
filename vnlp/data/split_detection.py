"""
split_detection.py
==================
Chia ảnh gốc + nhãn YOLO (dataset_CS406/raw/view*/{images,labels}) thành train/val/test
cho YOLO, ĐỒNG BỘ với split của recognition:

  * Ảnh nào có crop biển số trong recognition_split.csv -> cùng split với crop đó.
    => tập test detection chứa đúng các biển của tập test OCR, chạy pipeline end-to-end
       trên test không bị rò rỉ. Ảnh có nhiều biển thuộc nhiều split -> ưu tiên test > val > train.
  * View liên kết (config: split.linked_views, vd view5 = khung hình rộng của view1)
    -> cùng split với ảnh cùng tên ở view gốc.
  * Ảnh không có crop nào (crop bị reject...) -> chia bù để mỗi view gần đúng tỉ lệ.

Đầu ra:
  dataset/splits/detection/images/{train,val,test}/<view>_<idx>.jpg
  dataset/splits/detection/labels/{train,val,test}/<view>_<idx>.txt
  dataset/splits/detection/data.yaml      (không ghi `path` -> YOLO lấy thư mục chứa file)
  dataset/detection_split.csv              (image_id, split, view, source_image, n_boxes, split_source)

Chạy SAU vnlp/data/split_recognition.py:
  python -m vnlp.data.split_detection
"""

import csv
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from vnlp import config

SPLITS = ("train", "val", "test")
PRIORITY = {"test": 2, "val": 1, "train": 0}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_recognition_manifest(path: Path):
    """(view, raw_image_stem) -> set(split) của các crop sinh ra từ ảnh đó."""
    if not path.is_file():
        raise FileNotFoundError(f"Chưa có {path}. Chạy vnlp/data/split_recognition.py trước.")
    mapping = defaultdict(set)
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            stem = re.sub(r"__box\d+$", "", row["source_folder"])
            mapping[(row["view"], stem)].add(row["split"])
    return mapping


def read_yolo_label(path: Path):
    """Trả về list dòng hợp lệ 'cls cx cy w h' (bỏ dòng sai định dạng)."""
    lines, bad = [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            int(parts[0])
            values = [float(v) for v in parts[1:]]
            ok = len(values) == 4 and all(0.0 <= v <= 1.0 for v in values)
        except ValueError:
            ok = False
        if ok:
            lines.append(" ".join(parts))
        else:
            bad += 1
    return lines, bad


def collect_images(raw_dir: Path, views):
    items = []
    for view in views:
        image_dir, label_dir = raw_dir / view / "images", raw_dir / view / "labels"
        if not image_dir.is_dir():
            print(f"[{view}] bỏ qua: không có {image_dir}")
            continue
        for img in sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
            label = label_dir / f"{img.stem}.txt"
            if not label.is_file():
                print(f"[{view}] thiếu label: {img.name}")
                continue
            lines, bad = read_yolo_label(label)
            if bad:
                print(f"[{view}] {img.name}: bỏ {bad} dòng nhãn sai định dạng")
            items.append({"view": view, "image": img, "stem": img.stem, "lines": lines})
    return items


def assign_splits(items, recog, linked, ratios, seed):
    rng = random.Random(seed)
    stats = Counter()

    # 1) theo recognition
    for it in items:
        if it["view"] in linked:
            continue
        splits = recog.get((it["view"], it["stem"]))
        if splits:
            it["split"] = max(splits, key=PRIORITY.get)
            it["split_source"] = "recognition" if len(splits) == 1 else "recognition_conflict"
            stats[it["split_source"]] += 1

    # 2) ảnh còn lại của view thường -> bù cho đủ tỉ lệ trong từng view
    by_view = defaultdict(list)
    for it in items:
        if it["view"] not in linked:
            by_view[it["view"]].append(it)
    for view, group in by_view.items():
        current = Counter(it["split"] for it in group if "split" in it)
        free = [it for it in group if "split" not in it]
        rng.shuffle(free)
        for it in free:
            best = max(SPLITS, key=lambda s: ratios[s] * len(group) - current[s])
            it["split"], it["split_source"] = best, "balanced"
            current[best] += 1
            stats["balanced"] += 1

    # 3) view liên kết -> theo ảnh cùng tên ở view gốc
    parent_split = {(it["view"], it["stem"]): it["split"] for it in items if "split" in it}
    for it in items:
        if it["view"] not in linked:
            continue
        split = parent_split.get((linked[it["view"]], it["stem"]))
        if split:
            it["split"], it["split_source"] = split, f"linked:{linked[it['view']]}"
        else:
            it["split"], it["split_source"] = rng.choices(SPLITS, weights=[ratios[s] for s in SPLITS])[0], "random"
        stats[it["split_source"].split(":")[0]] += 1
    return stats


def write(items, out_dir: Path, manifest: Path):
    for sub in ("images", "labels"):
        for split in SPLITS:
            d = out_dir / sub / split
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)

    counters, rows = Counter(), []
    for it in items:
        counters[it["view"]] += 1
        image_id = f"{it['view']}_{counters[it['view']]:05d}"
        shutil.copy2(it["image"], out_dir / "images" / it["split"] / f"{image_id}{it['image'].suffix.lower()}")
        (out_dir / "labels" / it["split"] / f"{image_id}.txt").write_text(
            "\n".join(it["lines"]) + ("\n" if it["lines"] else ""), encoding="utf-8")
        rows.append({"image_id": image_id, "split": it["split"], "view": it["view"],
                     "source_image": it["image"].name, "n_boxes": len(it["lines"]),
                     "split_source": it["split_source"]})

    (out_dir / "data.yaml").write_text(
        "# Sinh bởi vnlp/data/split_detection.py – không có `path` nên YOLO dùng thư mục chứa file này\n"
        "train: images/train\nval: images/val\ntest: images/test\n\nnames:\n  0: license_plate\n",
        encoding="utf-8")

    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def report(items, stats):
    views = sorted({it["view"] for it in items})
    imgs = Counter((it["view"], it["split"]) for it in items)
    boxes = Counter()
    for it in items:
        boxes[(it["view"], it["split"])] += len(it["lines"])
    print(f"\n{'view':<8}" + "".join(f"{s:>14}" for s in SPLITS) + f"{'total':>14}")
    for v in views + ["ALL"]:
        cell = lambda s, c: sum(n for (vv, ss), n in c.items() if ss == s and (v == "ALL" or vv == v))  # noqa: E731
        parts = [f"{cell(s, imgs)} ({cell(s, boxes)}b)" for s in SPLITS]
        total = sum(cell(s, imgs) for s in SPLITS)
        print(f"{v:<8}" + "".join(f"{p:>14}" for p in parts) + f"{total:>14}")
    print("(số ảnh, b = số box)")
    print("Nguồn quyết định split:", dict(stats))


def main():
    cfg = config.CFG["split"]
    ratios = dict(zip(SPLITS, (r / sum(cfg["ratios"]) for r in cfg["ratios"])))
    linked = cfg.get("linked_views") or {}
    views = cfg["detection_views"]

    recog = read_recognition_manifest(config.RECOG_MANIFEST)
    items = collect_images(config.RAW_DIR, views)
    print(f"Đọc được {len(items)} ảnh từ {config.RAW_DIR} ({', '.join(views)})")
    stats = assign_splits(items, recog, linked, ratios, config.SEED)
    write(items, config.DET_SPLIT_DIR, config.DET_MANIFEST)
    report(items, stats)
    print(f"\nĐã ghi: {config.DET_SPLIT_DIR}\nManifest: {config.DET_MANIFEST}")


if __name__ == "__main__":
    main()
