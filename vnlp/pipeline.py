"""
pipeline.py
===========
Pipeline end-to-end: ảnh gốc -> YOLO -> crop -> preprocess -> OCR (PARSeq/CRNN) -> vn_plate.

    from vnlp.pipeline import PlatePipeline, evaluate_end_to_end
    pipe = PlatePipeline(ocr="parseq")
    pipe.predict(image_bgr)              # list dict {box, conf, raw, text, valid}
    df_img, summary = evaluate_end_to_end(pipe)   # trên tập test detection

Ground truth cho end-to-end lấy từ 2 manifest: ảnh detection -> các crop recognition cùng ảnh gốc.
"""

import re
from pathlib import Path

import pandas as pd
import torch

from vnlp import config
from vnlp.utils.io import imread, normalize_label
from vnlp.recognition.preprocess import preprocess_plate
from vnlp.recognition.postprocess import postprocess_ocr


class PlatePipeline:
    def __init__(self, yolo_weights=None, ocr="parseq", ocr_weights=None, conf=None, device=None):
        from ultralytics import YOLO

        self.device = device or config.get_device()
        self.conf = config.pick(conf, "detection", "conf")
        self.detector = YOLO(str(yolo_weights or config.YOLO_WEIGHTS))
        self.ocr_name = ocr
        if ocr == "parseq":
            from vnlp.recognition import parseq as ocr_mod
            self._model = ocr_mod.load_model(ocr_weights or config.PARSEQ_WEIGHTS, self.device)
            self._tf = ocr_mod.build_transform(augment=False)
            self._decode = lambda x: ocr_mod.decode(self._model, x)
        elif ocr == "crnn":
            from vnlp.recognition import crnn as ocr_mod
            from torchvision import transforms
            self._model = ocr_mod.CRNN().to(self.device)
            self._model.load_state_dict(torch.load(ocr_weights or config.CRNN_WEIGHTS,
                                                   map_location=self.device, weights_only=True))
            self._model.eval()
            self._tf = transforms.Compose([transforms.ToPILImage(),
                                           transforms.Resize((config.IMG_H, config.IMG_W)),
                                           transforms.ToTensor()])
            self._decode = lambda x: ocr_mod.decode_predictions(self._model(x).argmax(2))
        else:
            raise ValueError("ocr phải là 'parseq' hoặc 'crnn'")

    def read_plate(self, crop_bgr) -> str:
        gray = preprocess_plate(crop_bgr)
        if self.ocr_name == "parseq":
            import cv2
            gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        with torch.no_grad():
            return self._decode(self._tf(gray).unsqueeze(0).to(self.device))[0]

    def predict(self, image_bgr):
        result = self.detector(image_bgr, conf=self.conf, verbose=False, device=self.device)[0]
        h, w = image_bgr.shape[:2]
        plates = []
        for (x1, y1, x2, y2), score in zip(result.boxes.xyxy.cpu().int().tolist(), result.boxes.conf.cpu().tolist()):
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            crop = image_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            raw = self.read_plate(crop)
            post = postprocess_ocr(raw, return_two_lines=False)
            plates.append({"box": (x1, y1, x2, y2), "conf": round(score, 3), "raw": raw,
                           "text": normalize_label(post["formatted"]), "valid": post["valid"]})
        return plates


def ground_truth_by_image(split="test"):
    """image_id (detection) -> (đường dẫn ảnh, list biển GT). Chỉ ảnh có crop đã gán nhãn."""
    det = pd.read_csv(config.DET_MANIFEST)
    rec = pd.read_csv(config.RECOG_MANIFEST)
    rec["stem"] = rec["source_folder"].map(lambda s: re.sub(r"__box\d+$", "", s))
    plates = rec.groupby(["view", "stem"])["plate"].apply(list).to_dict()

    out = {}
    for row in det[det["split"] == split].itertuples():
        gt = plates.get((row.view, Path(row.source_image).stem))
        if gt:
            image = next((config.DET_SPLIT_DIR / "images" / split).glob(f"{row.image_id}.*"))
            out[row.image_id] = (image, gt)
    return out


def evaluate_end_to_end(pipe: PlatePipeline, split="test", image_ids=None, out_dir=None):
    """
    Mỗi biển GT tính là đúng nếu có 1 dự đoán (sau vn_plate) trùng khớp hoàn toàn.
    image_ids: chỉ đánh giá các ảnh này (None = toàn bộ split).
    Trả về (df theo ảnh, df tổng hợp theo view).
    """
    rows = []
    items = [(k, v) for k, v in ground_truth_by_image(split).items() if image_ids is None or k in set(image_ids)]
    for image_id, (path, gt_plates) in items:
        preds = pipe.predict(imread(path))
        texts = [p["text"] for p in preds]
        hit = sum(g in texts for g in gt_plates)
        rows.append({"image_id": image_id, "view": image_id.split("_")[0], "gt": "|".join(gt_plates),
                     "pred": "|".join(texts), "n_gt": len(gt_plates), "n_pred": len(preds), "n_correct": hit})
    df = pd.DataFrame(rows)

    def agg(name, part):
        return {"view": name, "images": len(part), "plates": int(part["n_gt"].sum()),
                "e2e_plate_acc": round(100 * part["n_correct"].sum() / max(part["n_gt"].sum(), 1), 2),
                "images_all_correct": round(100 * (part["n_correct"] == part["n_gt"]).mean(), 2)}

    summary = pd.DataFrame([agg("ALL", df)] + [agg(v, g) for v, g in sorted(df.groupby("view"))]) if len(df) else pd.DataFrame()
    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / f"e2e_{pipe.ocr_name}_{split}_images.csv", index=False, encoding="utf-8")
        summary.to_csv(out_dir / f"e2e_{pipe.ocr_name}_{split}_summary.csv", index=False, encoding="utf-8")
    return df, summary
