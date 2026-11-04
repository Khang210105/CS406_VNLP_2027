import os
import sys
import cv2
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt', quiet=True)

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# --- IMPORT THƯ VIỆN PARSEQ CHÍNH THỨC ---
from strhub.models.parseq.system import PARSeq
from preprocess import preprocess_plate

# =====================================================
# KIỂM TRA PHIÊN BẢN VÀ CẤU HÌNH CUDA CHO GTX 1650
# =====================================================
print("="*50)
print("🔍 KIỂM TRA CẤU HÌNH HỆ THỐNG & CARD ĐỒ HỌA:")
print(f"  - Python Version: {sys.version.split()[0]}")
print(f"  - PyTorch Version: {torch.__version__}")
print(f"  - PyTorch Lightning Version: {pl.__version__}")
print(f"  - CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"  - 🚀 Đang nhận Card: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True
else:
    print("  - ❌ CẢNH BÁO: PyTorch chưa nhận được CUDA!")
print("="*50)

SEED = 42
pl.seed_everything(SEED)

# =====================================================
# CONFIG
# =====================================================
TRAIN_DIR = r"C:\Users\MrBeast\Desktop\CS338_PR\project\ocr_split\train"
VALID_DIR = r"C:\Users\MrBeast\Desktop\CS338_PR\project\ocr_split\valid"
OUTPUT_DIR = r"C:\Users\MrBeast\Desktop\CS338_PR\project\model_2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 100       
IMG_W = 320
IMG_H = 128
CHARS = "0123456789ABCDEFGHKLMNPRSTUVXYZĐ" # Q W I O J 

# =====================================================
# DATASET
# =====================================================
class PARSeqDataset(Dataset):
    def __init__(self, root_dir, augment=True):
        self.samples = []
        if not os.path.exists(root_dir):
            print(f"❌ Không tìm thấy thư mục: {root_dir}")
            return
            
        folders = os.listdir(root_dir)
        for folder in folders:
            sample_dir = os.path.join(root_dir, folder)
            if not os.path.isdir(sample_dir):
                continue

            image_path = os.path.join(sample_dir, "image.jpg")
            label_path = os.path.join(sample_dir, "label.txt")

            if not os.path.exists(image_path) or not os.path.exists(label_path):
                continue

            with open(label_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
                text = text.replace("-", "").replace(".", "").replace("\n", "").replace(" ", "")

            if len(text) == 0:
                continue

            self.samples.append((image_path, text))

        if augment:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((IMG_H, IMG_W)),
                # transforms.RandomRotation(3),
                transforms.RandomRotation(5),
                # transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ColorJitter( ),
                transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((IMG_H, IMG_W)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, text = self.samples[idx]
        image = cv2.imread(image_path)

        if image is None:
            processed_rgb = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
        else:
            processed_gray = preprocess_plate(image) 
            if len(processed_gray.shape) == 2:
                processed_rgb = cv2.cvtColor(processed_gray, cv2.COLOR_GRAY2RGB)
            else:
                processed_rgb = cv2.cvtColor(processed_gray, cv2.COLOR_BGR2RGB)

        image_tensor = self.transform(processed_rgb)
        return image_tensor, text

def collate_fn(batch):
    images = []
    texts = []
    for image, text in batch:
        images.append(image)
        texts.append(text)
    images = torch.stack(images)
    return images, texts

# =====================================================
# CALLBACK TÙY CHỈNH: IN MẪU DỰ ĐOÁN & LƯU/VẼ QUÁ TRÌNH HỌC
# =====================================================
class TrackingAndVisualCallback(pl.Callback):
    def __init__(self, output_dir):
        super().__init__()
        self.output_dir = output_dir
        self.history = []
        self.history_file = os.path.join(output_dir, "history_parseq.csv")

    def on_validation_epoch_end(self, trainer, pl_module):
        # 1. PHẦN IN MẪU DỰ ĐOÁN THỰC TẾ TRÊN TERMINAL
        val_loaders = trainer.val_dataloaders
        if not val_loaders:
            return
        
        val_loader = val_loaders[0] if isinstance(val_loaders, list) else val_loaders
        try:
            batch = next(iter(val_loader))
        except StopIteration:
            return
            
        images, texts = batch
        images = images.to(pl_module.device)
        
        pl_module.eval()
        with torch.no_grad():
            preds = pl_module.forward(images)
            decode_output = pl_module.tokenizer.decode(preds)
            
            if isinstance(decode_output, tuple):
                decoded_strings = decode_output[0]
            else:
                decoded_strings = decode_output
            
        print(f"\n--- [Epoch {trainer.current_epoch + 1}] Mẫu dự đoán thực tế PARSeq tập Val ---")
        for i in range(min(5, len(texts))):
            raw_str = str(decoded_strings[i])
            p_clean = raw_str.split('[E]')[0].strip()
            print(f"Nhãn gốc: {texts[i]:<12} | PARSeq đoán: {p_clean}")
        print("--------------------------------------------------")

        # 2. PHẦN THU THẬP METRICS SAU MỖI EPOCH (CHỈ LẤY KHI BẮT ĐẦU CHẠY THẬT, BỎ QUA SANITY CHECK)
        if trainer.sanity_checking:
            return

        metrics = trainer.callback_metrics
        # Trích xuất các giá trị loss và accuracy nội tại của PARSeq từ bộ nhớ Lightning
        train_loss = metrics.get('train_loss', torch.tensor(float('nan'))).item()
        val_loss = metrics.get('val_loss', torch.tensor(float('nan'))).item()
        # Chú ý: PARSeq của strhub định nghĩa độ chính xác từ khóa là 'val_ned' hoặc 'val_accuracy' tùy phiên bản cấu hình mã nguồn
        val_acc = metrics.get('val_accuracy', metrics.get('val_ned', torch.tensor(0.0))).item()
        if val_acc <= 1.0: # Đổi từ hệ thập phân sang % nếu cần
            val_acc *= 100

        self.history.append({
            "epoch": trainer.current_epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc
        })
        
        # Ghi đè cập nhật liên tục ra file CSV phòng trường hợp mất điện/tắt ngang
        pd.DataFrame(self.history).to_csv(self.history_file, index=False)

    def on_train_end(self, trainer, pl_module):
        # 3. TỰ ĐỘNG VẼ BIỂU ĐỒ DASHBOARD KHI KẾT THÚC QUÁ TRÌNH TRAINING
        if not self.history:
            print("⚠️ Không có dữ liệu lịch sử để vẽ biểu đồ.")
            return

        print("\n📊 Huấn luyện hoàn tất! Đang tiến hành xuất biểu đồ...")
        df = pd.DataFrame(self.history)
        plt.figure(figsize=(14, 6))

        # Biểu đồ Loss
        plt.subplot(1, 2, 1)
        if "train_loss" in df.columns and not df["train_loss"].isna().all():
            plt.plot(df["epoch"], df["train_loss"], label="Train Loss", color="#8884d8", linewidth=2)
        plt.plot(df["epoch"], df["val_loss"], label="Validation Loss", color="#82ca9d", linewidth=2)
        plt.title("Mô hình PARSeq - Tiến trình Loss", fontsize=12, fontweight='bold')
        plt.xlabel("Epochs")
        plt.ylabel("Loss Value")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()

        # Biểu đồ Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(df["epoch"], df["val_acc"], label="Validation Accuracy", color="#413ea0", linewidth=2)
        plt.title("Mô hình PARSeq - Độ chính xác (Accuracy)", fontsize=12, fontweight='bold')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (%)")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()

        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, "parseq_training_results.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"🎉 File lịch sử học tập CSV và biểu đồ đồ thị đã được lưu thành công tại: {self.output_dir}")

# =====================================================
# MAIN EXECUTION
# =====================================================
if __name__ == "__main__":
    train_dataset = PARSeqDataset(TRAIN_DIR, augment=True)
    val_dataset   = PARSeqDataset(VALID_DIR, augment=False)

    print(f"📊 Tập Train: {len(train_dataset)} mẫu")
    print(f"📊 Tập Valid: {len(val_dataset)} mẫu")

    num_workers = 0 if os.name == 'nt' else 4

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn
    )

    print("\n🧠 Khởi tạo mô hình và cấu hình Lightning Trainer...")
    model = PARSeq(
        charset_train=CHARS,         
        charset_test=CHARS,          
        max_label_length=12,         
        batch_size=BATCH_SIZE,       
        lr=0.0005,                   
        warmup_pct=0.07,             
        weight_decay=1e-4,           
        img_size=[IMG_H, IMG_W],     
        patch_size=[16, 16],         
        embed_dim=384,               
        enc_num_heads=12,            
        enc_mlp_ratio=4,           
        enc_depth=12,                
        dec_num_heads=12,            
        dec_mlp_ratio=4,           
        dec_depth=1,                 
        perm_num=6,                  
        perm_forward=True,           
        perm_mirrored=True,          
        decode_ar=True,              
        refine_iters=1,              
        dropout=0.1                  
    )

    # Cấu hình tự động lưu checkpoint có val_loss tối ưu nhất
    checkpoint_callback = ModelCheckpoint(
        dirpath=OUTPUT_DIR,
        filename="parseq_model",
        monitor="val_loss",
        mode="min",
        save_top_k=1
    )

    # Đưa mô hình lên xử lý bằng GPU CUDA và tích hợp Callback theo dõi lịch sử mới
    tracking_callback = TrackingAndVisualCallback(OUTPUT_DIR)

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_callback, tracking_callback],
        enable_progress_bar=True
    )

    print("🚀 Bắt đầu quá trình huấn luyện bằng PyTorch Lightning...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(f"✅ Huấn luyện hoàn tất! Trọng số mô hình tốt nhất (.ckpt) đã lưu tại thư mục: {OUTPUT_DIR}")