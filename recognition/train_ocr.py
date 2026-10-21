import os
import cv2
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from preprocess import preprocess_plate

SEED = 42

random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# =====================================================
# CONFIG
# =====================================================

TRAIN_DIR = r"C:\Users\MrBeast\Desktop\CS338_PR\project\ocr_split\train"
VALID_DIR = r"C:\Users\MrBeast\Desktop\CS338_PR\project\ocr_split\valid"

BATCH_SIZE = 16
EPOCHS = 100

IMG_W = 320
IMG_H = 128

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# CHARSET
# =====================================================

CHARS = "0123456789ABCDEFGHKLMNPRSTUVXYZĐ" # Q W I O J 

char_to_idx = {c: i + 1 for i, c in enumerate(CHARS)}
idx_to_char = {i + 1: c for i, c in enumerate(CHARS)}

NUM_CLASSES = len(CHARS) + 1  # + blank CTC

def decode_predictions(preds):
    """
    Dịch ngược tensor dự đoán (chứa index) thành chuỗi text.
    Áp dụng luật Greedy CTC: Xóa ký tự blank (0) và các ký tự lặp liên tiếp.
    """
    texts = []
    for i in range(preds.size(0)):
        pred = preds[i].tolist()
        text = ""
        prev = -1
        for p in pred:
            # 0 là index của blank do ta thiết lập CTCLoss(blank=0)
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

        self.samples = []

        folders = os.listdir(root_dir)

        for folder in folders:

            sample_dir = os.path.join(root_dir, folder)

            # BUG FIX 1: Bỏ qua nếu không phải thư mục
            if not os.path.isdir(sample_dir):
                continue

            image_path = os.path.join(sample_dir, "image.jpg")
            label_path = os.path.join(sample_dir, "label.txt")

            if not os.path.exists(image_path):
                continue

            if not os.path.exists(label_path):
                continue

            with open(label_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
                text = text.replace("-", "")
                text = text.replace(".", "")
                text = text.replace("\n", "")
                text = text.replace(" ", "")

            # BUG FIX 2: Bỏ qua sample có label rỗng để tránh lỗi CTCLoss
            if len(text) == 0:
                continue

            self.samples.append((image_path, text))

        # BUG FIX 3: Chỉ áp dụng augmentation khi train, dùng flag để kiểm soát
        # (Hiện tại transform luôn có RandomRotation & ColorJitter kể cả val,
        #  nhưng vì val_dataset dùng chung dataset object nên đây là hạn chế cần lưu ý)
        if augment:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((IMG_H, IMG_W)),
                transforms.RandomRotation(3),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor()
            ])
        else:
            self.transform = transforms.Compose([  # val/test không augment
                transforms.ToPILImage(),
                transforms.Resize((IMG_H, IMG_W)),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.samples)

    def encode(self, text):
        # Mã hóa thuần túy ký tự sang index dựa trên CHARSET.
        # nn.CTCLoss của PyTorch sẽ tự xử lý các ký tự trùng lặp dựa trên thuật toán dynamic programming.
        encoded = [char_to_idx[c] for c in text if c in char_to_idx]
        return torch.tensor(encoded, dtype=torch.long)

    def __getitem__(self, idx):
        image_path, text = self.samples[idx]

        # 1. Đọc ảnh thô từ ổ cứng (Bất kể ảnh to hay nhỏ của bạn)
        image = cv2.imread(image_path)

        if image is None:
            # Dự phòng nếu ảnh lỗi, tạo ma trận đen để code không sập
            processed = np.zeros((128, 320), dtype=np.uint8)
        else:
            # 2. CHẠY TIỀN XỬ LÝ NÂNG CAO TRÊN RAM (Xoay thẳng, khử nhiễu, làm nét...)
            processed = preprocess_plate(image) # Đầu ra đã là ảnh xám Grayscale

        # 3. Đưa qua transform để Resize về (128, 320) và chuyển thành Tensor cho PyTorch
        image = self.transform(processed)

        label = self.encode(text)

        return image, label, text

# =====================================================
# COLLATE
# =====================================================

def collate_fn(batch):
    images = []
    labels = []
    label_lengths = []
    texts = []

    for image, label, text in batch:  # 👈 Đồng bộ hứng đúng 3 tham số từ __getitem__
        images.append(image)
        labels.extend(label.tolist())
        label_lengths.append(len(label))  # Tự tính độ dài nhãn ở đây
        texts.append(text)

    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=torch.long)

    return images, labels, label_lengths, texts

# =====================================================
# MODEL
# =====================================================

class CRNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),           # H: 128->464, W: 320->160

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),           # H: 64->32, W: 160->80

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),      # H: 32->16, W: 80->80

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2, 1))       # H: 16->8, W: 80->80
        )

        # H sau CNN = 8, C = 512 => input_size = 512 * 8 = 4096
        self.rnn = nn.LSTM(
            input_size=512 * 8,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            dropout=0.3,
            batch_first=True
        )

        # bidirectional => hidden_size * 2 = 512
        self.fc = nn.Linear(512, NUM_CLASSES)

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()
        # Reshape: (B, C, H, W) -> (B, W, C*H) để LSTM đọc theo chiều ngang
        x = x.permute(0, 3, 1, 2)   # (B, W, C, H)
        x = x.reshape(b, w, c * h)   # (B, W, C*H)
        x, _ = self.rnn(x)           # (B, W, 512)
        x = self.fc(x)               # (B, W, NUM_CLASSES)
        x = x.log_softmax(2)
        return x

if __name__ == "__main__":
    # =====================================================
    # TRAIN
    # =====================================================

    # Load riêng từng split thay vì dùng random_split
    train_dataset = OCRDataset(TRAIN_DIR, augment=True)
    val_dataset   = OCRDataset(VALID_DIR,augment=False)

    print(f"Train: {len(train_dataset)} samples")
    print(f"Valid: {len(val_dataset)} samples")

    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,      # 👈 Chia cho 2 lõi CPU cùng xử lý ảnh song song
        pin_memory=True,     # 👈 Đẩy nhanh tốc độ chuyển dữ liệu từ RAM lên VRAM của GPU
        prefetch_factor=2,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=4,      # 👈 Sửa tương tự cho tập val
        pin_memory=True,
        prefetch_factor=2,
        collate_fn=collate_fn
    )

    model = CRNN().to(DEVICE)

    # zero_infinity=True: tránh loss = inf khi input_length < target_length
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-4)

    # =====================================================
    # LOOP
    # =====================================================

    # List lưu lịch sử để vẽ đồ thị
    history = []
    history_file = r"C:\Users\MrBeast\Desktop\CS338_PR\project\model\history_ocr.csv"

    print("🚀 Bắt đầu quá trình huấn luyện mô hình CRNN - OCR...")
    best_val = float('inf')  # Khởi tạo giá trị vô cùng lớn ban đầu
    for epoch in range(EPOCHS):
        # ------------------ PHẦN TRAIN ------------------
        model.train()
        total_train_loss = 0
        correct_train_plates = 0
        total_train_samples = 0
        
        loop = tqdm(train_loader)
        for batch_idx, (images, labels, label_lengths, texts) in enumerate(loop):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            outputs_ctc = outputs.permute(1, 0, 2)

            input_lengths = torch.full(size=(images.size(0),), fill_value=outputs_ctc.size(0), dtype=torch.long, device=DEVICE)
            target_lengths = torch.tensor(label_lengths, dtype=torch.long, device=DEVICE)

            loss = criterion(outputs_ctc, labels, input_lengths, target_lengths)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_train_loss += loss.item()
            
            # Tính toán Accuracy tập Train nhanh
            preds = outputs.argmax(2)
            decoded_texts = decode_predictions(preds)
            for pred_text, gt_text in zip(decoded_texts, texts):
                if pred_text == gt_text:
                    correct_train_plates += 1
            total_train_samples += len(texts)

            loop.set_description(f"Epoch {epoch+1}/{EPOCHS}")
            loop.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)
        train_acc = (correct_train_plates / total_train_samples) * 100

        # ------------------ PHẦN VALIDATION ------------------
        model.eval()
        total_val_loss = 0
        correct_val_plates = 0
        total_val_samples = 0
        printed_sample = False

        with torch.no_grad():
            for images, labels, label_lengths, texts in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)
                
                # In mẫu dự đoán thực tế của batch đầu tiên để theo dõi tiến độ chữ ra
                if not printed_sample:
                    preds = outputs.argmax(2)
                    decoded_texts = decode_predictions(preds)
                    print(f"\n--- [Epoch {epoch+1}] Mẫu dự đoán thực tế tập Val ---")
                    for i in range(min(5, len(texts))):
                        print(f"Nhãn gốc: {texts[i]:<12} | Model đoán: {decoded_texts[i]}")
                    print("--------------------------------------------------")
                    printed_sample = True

                outputs_ctc = outputs.permute(1, 0, 2)
                input_lengths = torch.full((images.size(0),), outputs_ctc.size(0), dtype=torch.long)
                target_lengths = torch.tensor(label_lengths, dtype=torch.long)

                loss = criterion(outputs_ctc, labels, input_lengths, target_lengths)
                total_val_loss += loss.item()

                # Tính toán Accuracy tập Val
                preds = outputs.argmax(2)
                decoded_texts = decode_predictions(preds)
                for pred_text, gt_text in zip(decoded_texts, texts):
                    if pred_text == gt_text:
                        correct_val_plates += 1
                total_val_samples += len(texts)

        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = (correct_val_plates / total_val_samples) * 100

        print(f"📈 [Kết quả] Train Loss: {avg_train_loss:.4f} - Train Acc: {train_acc:.2f}% | Val Loss: {avg_val_loss:.4f} - Val Acc: {val_acc:.2f}%")

        # Lưu lịch sử epoch hiện tại
        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_acc": train_acc,
            "val_loss": avg_val_loss,
            "val_acc": val_acc
        })
        
        # Ghi dự phòng ra file CSV định kỳ
        pd.DataFrame(history).to_csv(history_file, index=False)

        # ------------------ LƯU MODEL BEST ------------------
        if avg_val_loss < best_val:
            best_val = avg_val_loss
            torch.save(model.state_dict(), r"C:\Users\MrBeast\Desktop\CS338_PR\project\model\ocr_model.pth")
            print(f"✅ Đã cập nhật và lưu file trọng số tốt nhất tại Epoch {epoch+1}!")

    # =====================================================================
    # TỰ ĐỘNG VẼ BIỂU ĐỒ DASHBOARD KHI KẾT THÚC HUẤN LUYỆN (YOLO STYLE)
    # =====================================================================
    print("\n📊 Huấn luyện hoàn tất! Đang tiến hành xuất biểu đồ Loss và Accuracy...")
    df = pd.DataFrame(history)

    plt.figure(figsize=(14, 6))

    # 1. Đồ thị Loss (Bên trái)
    plt.subplot(1, 2, 1)
    plt.plot(df["epoch"], df["train_loss"], label="Train CTC Loss", color="#8884d8", linewidth=2)
    plt.plot(df["epoch"], df["val_loss"], label="Validation CTC Loss", color="#82ca9d", linewidth=2)
    plt.title("Mô hình CRNN - Tiến trình CTC Loss", fontsize=12, fontweight='bold')
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    # 2. Đồ thị Accuracy (Bên phải)
    plt.subplot(1, 2, 2)
    plt.plot(df["epoch"], df["train_acc"], label="Train Plate Acc", color="#ff7300", linewidth=2)
    plt.plot(df["epoch"], df["val_acc"], label="Validation Plate Acc", color="#413ea0", linewidth=2)
    plt.title("Mô hình CRNN - Độ chính xác toàn biển (Plate Accuracy)", fontsize=12, fontweight='bold')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plot_path = r"C:\Users\MrBeast\Desktop\CS338_PR\project\model\ocr_training_results.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"🎉 Biểu đồ học tập đã được lưu thành công tại: {plot_path}")