"""
parseq.py
=========
PARSeq (strhub) nhận diện ký tự biển số: dataset, model, train() (PyTorch Lightning), evaluate().

Terminal (từ gốc repo, tham số lấy từ config.yaml -> parseq):
    python -m vnlp.recognition.parseq train
    python -m vnlp.recognition.parseq eval

Notebook:
    from vnlp.recognition import parseq
    ckpt = parseq.train(epochs=50)
    df_pred, df_metrics = parseq.evaluate(weights=ckpt)
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from strhub.models.parseq.system import PARSeq
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from vnlp import config
from vnlp.recognition.preprocess import preprocess_plate
from vnlp.utils.io import imread, load_samples
from vnlp.utils.metrics import print_eval, save_eval, summarize_predictions
from vnlp.utils.plotting import plot_history

# Charset / kích thước ảnh cố định lúc import (đổi ocr.* trong config.yaml thì restart kernel + train lại)
IMG_W, IMG_H, CHARS = config.IMG_W, config.IMG_H, config.CHARS
_NORMALIZE = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])


# =====================================================
# DATASET
# =====================================================
def build_transform(augment=False):
    """Transform cho ảnh RGB (numpy) đã qua preprocess_plate -> tensor chuẩn hóa."""
    steps = [transforms.ToPILImage(), transforms.Resize((IMG_H, IMG_W))]
    if augment:
        steps += [
            transforms.RandomRotation(5),
            transforms.ColorJitter(),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        ]
    steps += [transforms.ToTensor(), _NORMALIZE]
    return transforms.Compose(steps)


class PARSeqDataset(Dataset):
    def __init__(self, root_dir, augment=True):
        self.samples = load_samples(root_dir)
        self.transform = build_transform(augment)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, text, sample_id = self.samples[idx]
        image = imread(image_path)
        if image is None:
            rgb = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
        else:
            processed = preprocess_plate(image)
            code = cv2.COLOR_GRAY2RGB if processed.ndim == 2 else cv2.COLOR_BGR2RGB
            rgb = cv2.cvtColor(processed, code)
        return self.transform(rgb), text, sample_id


def collate_fn(batch):
    images, texts, ids = zip(*batch)
    return torch.stack(images), list(texts), list(ids)


class _LightningBatch(Dataset):
    """strhub PARSeq.training_step cần batch (images, labels) -> bỏ sample_id."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, text, _ = self.base[idx]
        return image, text


def _collate_lightning(batch):
    images, texts = zip(*batch)
    return torch.stack(images), list(texts)


def make_loader(root_dir, augment, batch_size, shuffle, num_workers=None, with_ids=False):
    num_workers = config.default_num_workers() if num_workers is None else num_workers
    dataset = PARSeqDataset(root_dir, augment=augment)
    if not with_ids:
        dataset, collate = _LightningBatch(dataset), _collate_lightning
    else:
        collate = collate_fn
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), collate_fn=collate)


# =====================================================
# MODEL
# =====================================================
def build_model(batch_size=None, lr=None):
    """Kiến trúc lấy từ config.yaml -> parseq.model."""
    kwargs = dict(config.CFG["parseq"]["model"])
    kwargs["warmup_pct"] = float(kwargs["warmup_pct"])
    kwargs["weight_decay"] = float(kwargs["weight_decay"])
    return PARSeq(
        charset_train=CHARS, charset_test=CHARS, max_label_length=config.MAX_LABEL_LENGTH,
        batch_size=config.pick(batch_size, "parseq", "batch_size"),
        lr=float(config.pick(lr, "parseq", "lr")),
        img_size=[IMG_H, IMG_W], **kwargs,
    )


def load_model(weights=None, device=None):
    device = device or config.get_device()
    weights = Path(weights or config.PARSEQ_WEIGHTS)
    if not weights.is_file():
        raise FileNotFoundError(f"Không tìm thấy trọng số PARSeq: {weights}. Hãy train trước.")
    model = build_model()
    checkpoint = torch.load(weights, map_location=device, weights_only=False)  # file tự train -> tin cậy
    model.load_state_dict(checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint)
    return model.to(device).eval()


def decode(model, images):
    """Trả về list chuỗi dự đoán cho batch ảnh đã lên đúng device."""
    output = model.tokenizer.decode(model(images))
    strings = output[0] if isinstance(output, tuple) else output
    return [str(s).split("[E]")[0].strip() for s in strings]


# =====================================================
# CALLBACK: in mẫu dự đoán + lưu history
# =====================================================
class TrackingCallback(pl.Callback):
    def __init__(self, output_dir):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.history = []

    def on_validation_epoch_end(self, trainer, pl_module):
        loaders = trainer.val_dataloaders
        loader = loaders[0] if isinstance(loaders, list) else loaders
        if loader is None:
            return
        images, texts = next(iter(loader))
        was_training = pl_module.training
        pl_module.eval()
        with torch.no_grad():
            preds = decode(pl_module, images.to(pl_module.device))
        pl_module.train(was_training)

        print(f"\n--- [Epoch {trainer.current_epoch + 1}] Mẫu dự đoán PARSeq (val) ---")
        for gt, pred in list(zip(texts, preds))[:5]:
            print(f"    GT: {gt:<12} Pred: {pred}")

        if trainer.sanity_checking:
            return
        m = trainer.callback_metrics
        nan = torch.tensor(float("nan"))
        val_acc = m.get("val_accuracy", m.get("val_NED", torch.tensor(0.0))).item()
        self.history.append({
            "epoch": trainer.current_epoch + 1,
            "train_loss": m.get("train_loss", m.get("loss", nan)).item(),
            "val_loss": m.get("val_loss", nan).item(),
            "val_acc": val_acc * 100 if val_acc <= 1.0 else val_acc,
        })
        pd.DataFrame(self.history).to_csv(self.output_dir / "history_parseq.csv", index=False)

    def on_train_end(self, trainer, pl_module):
        plot_history(self.history, self.output_dir / "parseq_training_results.png", "PARSeq")


def train(train_dir=None, val_dir=None, out_dir=None, epochs=None, batch_size=None, lr=None,
          num_workers=None, seed=None):
    """Train PARSeq, lưu checkpoint tốt nhất (val loss) vào out_dir. Trả về đường dẫn checkpoint.
    Tham số None -> lấy từ config.yaml (mục parseq / paths)."""
    train_dir = train_dir or config.RECOG_TRAIN_DIR
    val_dir = val_dir or config.RECOG_VAL_DIR
    out_dir = out_dir or config.PARSEQ_OUTPUT_DIR
    epochs = config.pick(epochs, "parseq", "epochs")
    batch_size = config.pick(batch_size, "parseq", "batch_size")
    lr = float(config.pick(lr, "parseq", "lr"))
    pl.seed_everything(config.pick(seed, "seed"))
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader = make_loader(train_dir, True, batch_size, True, num_workers)
    val_loader = make_loader(val_dir, False, batch_size, False, num_workers)
    print(f"Thiết bị: {config.get_device()} | Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

    checkpoint = ModelCheckpoint(dirpath=out_dir, filename="parseq_model",
                                 monitor="val_loss", mode="min", save_top_k=1)
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint, TrackingCallback(out_dir)],
        default_root_dir=out_dir,
    )
    trainer.fit(build_model(batch_size, lr), train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(f"✅ Xong. Checkpoint tốt nhất: {checkpoint.best_model_path}")
    return checkpoint.best_model_path


# =====================================================
# EVALUATE
# =====================================================
def evaluate(test_dir=None, weights=None, out_dir=None, batch_size=None, num_workers=None, verbose=True):
    """Plate Accuracy + CER trên tập test (tổng và theo view). Trả về (df_pred, df_metrics)."""
    test_dir = test_dir or config.RECOG_TEST_DIR
    out_dir = out_dir or config.PARSEQ_OUTPUT_DIR
    batch_size = config.pick(batch_size, "parseq", "batch_size")
    device = config.get_device()
    model = load_model(weights, device)
    loader = make_loader(test_dir, False, batch_size, False, num_workers, with_ids=True)

    records = []
    with torch.no_grad():
        for images, texts, ids in tqdm(loader, desc="Test PARSeq"):
            preds = decode(model, images.to(device))
            records += [{"sample_id": i, "gt": g, "pred": p} for i, g, p in zip(ids, texts, preds)]

    df_pred, df_metrics = summarize_predictions(records)
    save_eval(df_pred, df_metrics, out_dir, "parseq")
    if verbose:
        print_eval(df_pred, df_metrics, "PARSeq")
    return df_pred, df_metrics


if __name__ == "__main__":
    {"train": train, "eval": evaluate}[sys.argv[1] if len(sys.argv) > 1 else "train"]()
