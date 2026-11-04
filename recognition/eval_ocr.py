"""
test_ocr.py
===========
Đánh giá mô hình OCR đã huấn luyện trên tập TEST biệt lập.
Tính toán độ chính xác theo ký tự (Character Accuracy) và chính xác toàn bộ biển (Full Plate Accuracy).
"""

import os
import cv2
import torch
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Import module tiền xử lý nâng cao của bạn
from preprocess import preprocess_plate

# =====================================================
# CONFIG - ĐỒNG BỘ HOÀN TOÀN VỚI FILE TRAIN VÀ PREPROCESS
# =====================================================
TEST_DIR = r"C:\Users\MrBeast\Desktop\CS338_PR\project\ocr_split\test"
MODEL_PATH = r"C:\Users\MrBeast\Desktop\CS338_PR\project\model\ocr_model.pth"
# ERROR_DIR = r"C:\Users\MrBeast\Desktop\CS338_PR\project\ocr_error_images"

BATCH_SIZE = 16  # Tập test có thể để batch size lớn hơn chút cho nhanh
IMG_W = 320
IMG_H = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHARS = "0123456789ABCDEFGHKLMNPRSTUVXYZĐ" # Q W I O J 
char_to_idx = {c: i + 1 for i, c in enumerate(CHARS)}
idx_to_char = {i + 1: c for i, c in enumerate(CHARS)}
NUM_CLASSES = len(CHARS) + 1

# =====================================================
# DATASET CLASS FOR TEST (Tích hợp tiền xử lý nâng cao trên RAM)
# =====================================================
class OCRTestDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        if not os.path.exists(root_dir):
            print(f"❌ Không tìm thấy thư mục tập Test: {root_dir}")
            return

        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            if os.path.isdir(folder_path):
                img_path = os.path.join(folder_path, "image.jpg")
                txt_path = os.path.join(folder_path, "label.txt")
                
                if os.path.exists(img_path) and os.path.exists(txt_path):
                    with open(txt_path, "r", encoding="utf-8") as f:
                        text = f.read().strip().replace("-", "").replace(".", "").replace(" ", "").upper()
                    if len(text) > 0:
                        self.samples.append((img_path, text))

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, text = self.samples[idx]
        image = cv2.imread(image_path)
        
        if image is None:
            # Tạo ảnh đen dự phòng nếu ảnh lỗi
            processed = torch.zeros((1, IMG_H, IMG_W))
        else:
            # Chạy tiền xử lý nâng cao (Xoay thẳng, lọc nhiễu, làm nét)
            gray_processed = preprocess_plate(image)
            # Resize về kích thước chuẩn ma trận 320x128
            gray_resized = cv2.resize(gray_processed, (IMG_W, IMG_H))
            # Chuyển thành Tensor
            image = self.transform(gray_resized)
            
        return image, text

# =====================================================
# DECODE CTC OUTPUT (Greedy Decode)
# =====================================================
def decode_predictions(preds):
    decoded_texts = []
    for pred in preds:
        tokens = []
        prev_token = None
        for token in pred:
            token = token.item()
            if token != 0 and token != prev_token:  # 0 là ký tự blank của CTC
                tokens.append(idx_to_char[token])
            prev_token = token
        decoded_texts.append("".join(tokens))
    return decoded_texts

# =====================================================
# MODEL ARCHITECTURE (Phải trùng khớp 100% với train_ocr.py)
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

# =====================================================
# EVALUATION PIPELINE
# =====================================================
def evaluate_model():
    print("⏳ Đang khởi tạo và chuẩn bị tập Test dữ liệu...")
    test_dataset = OCRTestDataset(TEST_DIR)
    if len(test_dataset) == 0:
        return
        
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"📦 Tổng số lượng biển số trong tập TEST: {len(test_dataset)}")

    # Khởi tạo mô hình và load trọng số đã lưu
    model = CRNN().to(DEVICE)
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Không tìm thấy file trọng số model tại: {MODEL_PATH}. Bạn cần train trước!")
        return
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False))
    model.eval()
    print("✅ Đã load thành công file trọng số best ocr model!")

    total_characters = 0
    correct_characters = 0
    correct_plates = 0
    
    # Danh sách lưu các ca bị đoán sai để in mẫu phân tích lỗi
    failed_cases = []

    print("🚀 Đang tiến hành quét và đánh giá trên tập Test...")
    with torch.no_grad():
        for images, texts in tqdm(test_loader, desc="Testing"):
            images = images.to(DEVICE)
            outputs = model(images)
            
            preds = outputs.argmax(2)
            decoded_texts = decode_predictions(preds)

            for pred_text, ground_truth in zip(decoded_texts, texts):
                # 1. Tính toán độ chính xác toàn bộ biển số (Full Plate Accuracy)
                if pred_text == ground_truth:
                    correct_plates += 1
                else:
                    # if len(failed_cases) < 20:  # Lưu lại tối đa 5 ca sai để debug
                    failed_cases.append((ground_truth, pred_text))

                # 2. Tính toán độ chính xác theo từng ký tự (Character Accuracy)
                total_characters += len(ground_truth)
                
                # So sánh khớp từng vị trí ký tự (bằng thuật toán so ký tự cơ bản)
                match_chars = 0
                for c1, c2 in zip(pred_text, ground_truth):
                    if c1 == c2:
                        match_chars += 1
                correct_characters += match_chars

    # Tính toán phần trăm tỉ lệ
    plate_acc = (correct_plates / len(test_dataset)) * 100
    char_acc = (correct_characters / total_characters) * 100

    # =====================================================
    # IN KẾT QUẢ ĐÁNH GIÁ TỔNG QUÁT ĐỂ GỬI BÁO CÁO
    # =====================================================
    print("\n" + "="*60)
    print("📊 BÁO CÁO KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST (OCR)")
    print("="*60)
    print(f"🔹 Tổng số biển số đem ra thử nghiệm   : {len(test_dataset)}")
    print(f"🔹 Số biển số nhận diện đúng HOÀN TOÀN  : {correct_plates}/{len(test_dataset)}")
    print(f"🎯 ĐỘ CHÍNH XÁC TOÀN BIỂN (Plate Acc)  : {plate_acc:.2f}%")
    print(f"🎯 ĐỘ CHÍNH XÁC KÝ TỰ  (Char Acc)   : {char_acc:.2f}%")
    print("-" * 60)

    # 📥 XỬ LÝ 1: LƯU KẾT QUẢ VÀO FILE CSV
    ocr_data = {
        "Metric": ["Plate Accuracy (Chính xác toàn biển)", "Character Accuracy (Chính xác ký tự)"],
        "Value_Percent": [round(plate_acc, 2), round(char_acc, 2)],
        "Correct_Count": [correct_plates, correct_characters],
        "Total_Count": [len(test_dataset), total_characters]
    }
    df_ocr = pd.DataFrame(ocr_data)
    csv_path = "crnn_test_metrics.csv"
    df_ocr.to_csv(csv_path, index=False)
    print(f"💾 Đã lưu bảng số liệu test CRNN vào: {csv_path}")

    # 📊 XỬ LÝ 2: VẼ ĐỒ THỊ VÀ XUẤT FILE PNG

    plt.figure(figsize=(7, 5))
    metrics_labels = ["Plate Acc (Toàn biển)", "Char Acc (Từng ký tự)"]
    bars = plt.bar(metrics_labels, df_ocr["Value_Percent"], color=['#ff7300', '#413ea0'], width=0.4, edgecolor='black')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 2, f'{height}%', ha='center', va='bottom', fontweight='bold')

    plt.title("CRNN OCR Evaluation Result on Test Dataset", fontsize=12, fontweight='bold', pad=15)
    plt.ylabel("Độ chính xác (%)", fontsize=10)
    plt.ylim(0, 110)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    png_path = "crnn_test_metrics.png"
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"🖼️  Đã xuất đồ thị trực quan báo cáo CRNN: {png_path}\n")

if __name__ == "__main__":
    evaluate_model()