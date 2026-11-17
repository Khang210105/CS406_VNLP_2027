"""
crnn.py
=======
CRNN + CTC nhận diện ký tự biển số: model, dataset, train(), evaluate().

Terminal (từ gốc repo, tham số lấy từ config.yaml -> crnn):
    python -m vnlp.recognition.crnn train
    python -m vnlp.recognition.crnn eval

Notebook:
    from vnlp.recognition import crnn
    crnn.train(epochs=50)
    df_pred, df_metrics = crnn.evaluate()
"""

import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
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

char_to_idx = {c: i + 1 for i, c in enumerate(CHARS)}
idx_to_char = {i + 1: c for i, c in enumerate(CHARS)}
NUM_CLASSES = len(CHARS) + 1  # + blank CTC (index 0)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def decode_predictions(preds):
    """Greedy CTC: bỏ blank (0) và ký tự lặp liên tiếp. preds: (B, T) index."""
    texts = []
    for row in preds.tolist():
        text, prev = "", -1
        for p in row:
            if p != prev and p != 0:
                text += idx_to_char.get(p, "")
            prev = p
        texts.append(text)
    return texts


# =====================================================
# DATASET
# =====================================================
class OCRDataset(Dataset):
    def __init__(self, root_dir, augment=True):
        self.samples = load_samples(root_dir)
        if augment:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((IMG_H, IMG_W)),
                transforms.RandomRotation(3),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((IMG_H, IMG_W)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def encode(text):
        return torch.tensor([char_to_idx[c] for c in text if c in char_to_idx], dtype=torch.long)

    def __getitem__(self, idx):
        image_path, text, sample_id = self.samples[idx]
        image = imread(image_path)
        processed = np.zeros((IMG_H, IMG_W), dtype=np.uint8) if image is None else preprocess_plate(image)
        return self.transform(processed), self.encode(text), text, sample_id


def collate_fn(batch):
    images, labels, lengths, texts, ids = [], [], [], [], []
    for image, label, text, sample_id in batch:
        images.append(image)
        labels.extend(label.tolist())
        lengths.append(len(label))
        texts.append(text)
        ids.append(sample_id)
    return torch.stack(images), torch.tensor(labels, dtype=torch.long), lengths, texts, ids


# =====================================================
# MODEL
# =====================================================
class CRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                                    # 128x320 -> 64x160
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),                                    # -> 32x80
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d((2, 1)),                               # -> 16x80
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.MaxPool2d((2, 1)),                               # -> 8x80
        )
        self.rnn = nn.LSTM(input_size=512 * 8, hidden_size=256, num_layers=2,
                           bidirectional=True, dropout=0.3, batch_first=True)
        self.fc = nn.Linear(512, NUM_CLASSES)

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2).reshape(b, w, c * h)  # (B, W, C*H): đọc theo chiều ngang
        x, _ = self.rnn(x)
        return self.fc(x).log_softmax(2)                # (B, W, NUM_CLASSES)


def make_loader(root_dir, augment, batch_size, shuffle, num_workers=None):
    num_workers = config.default_num_workers() if num_workers is None else num_workers
    dataset = OCRDataset(root_dir, augment=augment)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), collate_fn=collate_fn)


def run_epoch(model, loader, criterion, device, optimizer=None, desc=""):
    """1 epoch train (có optimizer) hoặc val (không). Trả về (loss, acc%, mẫu dự đoán)."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total, examples = 0.0, 0, 0, []

    with torch.set_grad_enabled(training):
        for images, labels, lengths, texts, _ in tqdm(loader, desc=desc, leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs_ctc = outputs.permute(1, 0, 2)  # (T, B, C)
            input_lengths = torch.full((images.size(0),), outputs_ctc.size(0), dtype=torch.long)
            target_lengths = torch.tensor(lengths, dtype=torch.long)
            loss = criterion(outputs_ctc, labels, input_lengths, target_lengths)

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            total_loss += loss.item()
            decoded = decode_predictions(outputs.argmax(2))
            correct += sum(p == g for p, g in zip(decoded, texts))
            total += len(texts)
            if not examples:
                examples = list(zip(texts, decoded))[:5]

    return total_loss / max(len(loader), 1), 100 * correct / max(total, 1), examples


def train(train_dir=None, val_dir=None, out_dir=None, epochs=None, batch_size=None, lr=None,
          weight_decay=None, num_workers=None, seed=None):
    """Train CRNN, lưu best (theo val loss) + history.csv + biểu đồ vào out_dir. Trả về history.
    Tham số None -> lấy từ config.yaml (mục crnn / paths)."""
    train_dir = train_dir or config.RECOG_TRAIN_DIR
    val_dir = val_dir or config.RECOG_VAL_DIR
    out_dir = out_dir or config.CRNN_OUTPUT_DIR
    epochs = config.pick(epochs, "crnn", "epochs")
    batch_size = config.pick(batch_size, "crnn", "batch_size")
    lr = float(config.pick(lr, "crnn", "lr"))
    weight_decay = float(config.pick(weight_decay, "crnn", "weight_decay"))
    seed_everything(config.pick(seed, "seed"))
    device = config.get_device()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config.snapshot(out_dir)

    train_loader = make_loader(train_dir, True, batch_size, True, num_workers)
    val_loader = make_loader(val_dir, False, batch_size, False, num_workers)
    print(f"Thiết bị: {device} | Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

    model = CRNN().to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)  # zero_infinity: tránh inf khi T < len(label)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history, best_val = [], float("inf")
    weights_path = out_dir / "ocr_model.pth"
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc, _ = run_epoch(model, train_loader, criterion, device, optimizer, f"Epoch {epoch}/{epochs}")
        va_loss, va_acc, examples = run_epoch(model, val_loader, criterion, device, desc="Val")

        print(f"[Epoch {epoch}] Train loss {tr_loss:.4f} acc {tr_acc:.2f}% | Val loss {va_loss:.4f} acc {va_acc:.2f}%")
        for gt, pred in examples:
            print(f"    GT: {gt:<12} Pred: {pred}")

        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": va_loss, "val_acc": va_acc})
        pd.DataFrame(history).to_csv(out_dir / "history_ocr.csv", index=False)

        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), weights_path)
            print(f"    ✅ Lưu best model (val loss {va_loss:.4f}) -> {weights_path}")

    plot_history(history, out_dir / "ocr_training_results.png", "CRNN", acc_keys=("train_acc", "val_acc"))
    print(f"🎉 Xong. Kết quả lưu tại: {out_dir}")
    return history


# =====================================================
# EVALUATE
# =====================================================
def evaluate(test_dir=None, weights=None, out_dir=None, batch_size=None, num_workers=None, verbose=True):
    """Plate Accuracy + CER trên tập test (tổng và theo view). Trả về (df_pred, df_metrics)."""
    test_dir = test_dir or config.RECOG_TEST_DIR
    weights = Path(weights or config.CRNN_WEIGHTS)
    out_dir = out_dir or config.CRNN_OUTPUT_DIR
    batch_size = config.pick(batch_size, "crnn", "batch_size")
    if not weights.is_file():
        raise FileNotFoundError(f"Không tìm thấy trọng số CRNN: {weights}. Hãy train trước.")

    device = config.get_device()
    loader = make_loader(test_dir, False, batch_size, False, num_workers)
    model = CRNN().to(device)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.eval()

    records = []
    with torch.no_grad():
        for images, _, _, texts, ids in tqdm(loader, desc="Test CRNN"):
            preds = decode_predictions(model(images.to(device)).argmax(2))
            records += [{"sample_id": i, "gt": g, "pred": p} for i, g, p in zip(ids, texts, preds)]

    df_pred, df_metrics = summarize_predictions(records)
    save_eval(df_pred, df_metrics, out_dir, "crnn")
    if verbose:
        print_eval(df_pred, df_metrics, "CRNN")
    return df_pred, df_metrics


if __name__ == "__main__":
    {"train": train, "eval": evaluate}[sys.argv[1] if len(sys.argv) > 1 else "train"]()
