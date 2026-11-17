"""
split_recognition.py
====================
Chia tập crop biển số (processed_manual) thành train / val / test cho OCR.

Quy tắc chia:
  * Chia theo BIỂN SỐ (group split): một biển chỉ nằm ở đúng 1 split,
    kể cả khi biển đó xuất hiện ở nhiều view -> test luôn là biển chưa thấy.
  * protocol = stratified: phân tầng theo VIEW, mỗi view chia ~70/15/15 (theo số ảnh).
  * protocol = lodo: test = toàn bộ split.test_views; train/val từ các view còn lại.

Đầu ra (đúng format mà recognition/train_*.py, eval_*.py đang đọc):
  dataset/splits/recognition/{train,val,test}/<sample_id>/image.jpg + label.txt
  dataset/recognition_split.csv  (sample_id, split, view, plate, source_folder)

Tham số: config.yaml (paths.processed_dir, paths.data_dir, split.*, seed)
  python -m vnlp.data.split_recognition
Sau đó chạy vnlp/data/split_detection.py (detection dựa vào manifest này để chia cùng split).
"""

import csv
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from vnlp import config

SPLITS = ("train", "val", "test")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def load_samples(src: Path):
    samples = []
    for view_dir in sorted(p for p in src.iterdir() if p.is_dir() and p.name.startswith("view")):
        for folder in sorted(p for p in view_dir.iterdir() if p.is_dir()):
            label_path = folder / "label.txt"
            images = [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS]
            if not label_path.is_file() or len(images) != 1:
                print(f"[skip] {view_dir.name}/{folder.name}: thiếu label hoặc số ảnh != 1")
                continue
            plate = label_path.read_text(encoding="utf-8-sig").strip()
            if not plate:
                print(f"[skip] {view_dir.name}/{folder.name}: label rỗng")
                continue
            samples.append({"view": view_dir.name, "plate": plate,
                            "folder": folder.name, "image": images[0]})
    return samples


def assign_lodo(samples, ratios, seed, test_views):
    """
    Leave-one-domain-out: mọi mẫu của test_views -> test.
    Các view còn lại chia train/val theo biển số (tỉ lệ train:val), BỎ các mẫu có biển
    xuất hiện ở test_views (split = None) để test thật sự là biển + camera chưa thấy.
    """
    test_plates = {s["plate"] for s in samples if s["view"] in test_views}
    rest = []
    for s in samples:
        if s["view"] in test_views:
            s["split"] = "test"
        elif s["plate"] in test_plates:
            s["split"] = None
        else:
            rest.append(s)
    trainval = {"train": ratios["train"], "val": ratios["val"], "test": 0.0}
    total = trainval["train"] + trainval["val"]
    assign_splits(rest, {k: v / total for k, v in trainval.items()}, seed, splits=("train", "val"))


def assign_splits(samples, ratios, seed, splits=SPLITS):
    """Gán mỗi biển số vào 1 split; cân bằng số ảnh theo tỉ lệ trong từng view."""
    rng = random.Random(seed)
    plate_split = {}
    by_view = defaultdict(list)
    for s in samples:
        by_view[s["view"]].append(s)

    # View lớn trước để biển dùng chung giữa các view được quyết định ở view chiếm đa số.
    for view in sorted(by_view, key=lambda v: -len(by_view[v])):
        items = by_view[view]
        counts = Counter(s["plate"] for s in items)
        total = len(items)
        current = Counter()
        for plate, n in counts.items():
            if plate in plate_split:
                current[plate_split[plate]] += n

        free = [p for p in counts if p not in plate_split]
        rng.shuffle(free)
        # Biển nhiều ảnh xếp trước để greedy cân bằng tốt hơn (ổn định nhờ shuffle trước).
        free.sort(key=lambda p: -counts[p])
        for plate in free:
            deficit = {sp: ratios[sp] * total - current[sp] for sp in splits}
            best = max(splits, key=lambda sp: deficit[sp])
            plate_split[plate] = best
            current[best] += counts[plate]

    for s in samples:
        s["split"] = plate_split[s["plate"]]


def clean_dst(dst: Path):
    for sp in SPLITS:
        d = dst / sp
        d.mkdir(parents=True, exist_ok=True)
        for child in d.iterdir():
            if child.is_dir():
                shutil.rmtree(child)


def write_splits(samples, dst: Path, manifest: Path, quality: int):
    clean_dst(dst)
    counters = Counter()
    rows = []
    for s in samples:
        counters[s["view"]] += 1
        sample_id = f"{s['view']}_{counters[s['view']]:05d}"
        out_dir = dst / s["split"] / sample_id
        out_dir.mkdir(parents=True, exist_ok=True)
        with Image.open(s["image"]) as im:
            im.convert("RGB").save(out_dir / "image.jpg", quality=quality)
        (out_dir / "label.txt").write_text(s["plate"], encoding="utf-8")
        rows.append({"sample_id": sample_id, "split": s["split"], "view": s["view"],
                     "plate": s["plate"], "source_folder": s["folder"]})

    manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def report(samples):
    views = sorted({s["view"] for s in samples})
    table = Counter((s["view"], s["split"]) for s in samples)
    print(f"\n{'view':<8}" + "".join(f"{sp:>8}" for sp in SPLITS) + f"{'total':>8}")
    for v in views + ["ALL"]:
        cells = [sum(n for (vv, sp), n in table.items() if sp == s_ and (v == "ALL" or vv == v))
                 for s_ in SPLITS]
        print(f"{v:<8}" + "".join(f"{c:>8}" for c in cells) + f"{sum(cells):>8}")

    plates = defaultdict(set)
    for s in samples:
        plates[s["split"]].add(s["plate"])
    leak = (plates["train"] & plates["val"]) | (plates["train"] & plates["test"]) | (plates["val"] & plates["test"])
    print(f"\nSố biển khác nhau: " + ", ".join(f"{sp}={len(plates[sp])}" for sp in SPLITS))
    print(f"Biển bị trùng giữa các split: {len(leak)}")


def main():
    """Mọi tham số lấy từ config.yaml: paths.processed_dir, paths.data_dir, split.*, seed."""
    cfg = config.CFG["split"]
    raw_ratios = cfg["ratios"]
    ratios = dict(zip(SPLITS, (r / sum(raw_ratios) for r in raw_ratios)))

    samples = load_samples(config.PROCESSED_DIR)
    print(f"Đọc được {len(samples)} mẫu từ {config.PROCESSED_DIR} | protocol: {config.SPLIT_PROTOCOL}")
    if config.SPLIT_PROTOCOL == "lodo":
        print(f"Test = toàn bộ {config.TEST_VIEWS}")
        assign_lodo(samples, ratios, config.SEED, set(config.TEST_VIEWS))
        dropped = [s for s in samples if s["split"] is None]
        if dropped:
            print(f"Bỏ {len(dropped)} mẫu ở view train có biển trùng với view test: "
                  f"{dict(Counter(s['view'] for s in dropped))}")
        samples = [s for s in samples if s["split"] is not None]
    else:
        assign_splits(samples, ratios, config.SEED)
    write_splits(samples, config.RECOG_SPLIT_DIR, config.RECOG_MANIFEST, cfg["jpeg_quality"])
    report(samples)
    print(f"\nĐã ghi split vào {config.RECOG_SPLIT_DIR}\nManifest: {config.RECOG_MANIFEST}")


if __name__ == "__main__":
    main()
