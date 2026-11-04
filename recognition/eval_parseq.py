import os
import sys
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import editdistance

# --- IMPORT THƯ VIỆN PARSEQ CHÍNH THỨC ---
from strhub.models.parseq.system import PARSeq
from preprocess import preprocess_plate

# =====================================================
# CONFIG (Cấu hình chuẩn hóa tập TEST)
# =====================================================
TEST_DIR = r"C:\Users\MrBeast\Desktop\CS338_PR\project\ocr_split\test" 
MODEL_PATH = r"C:\Users\MrBeast\Desktop\CS338_PR\project\model\parseq_model.ckpt"
OUTPUT_DIR = r"C:\Users\MrBeast\Desktop\CS338_PR\project\model"

BATCH_SIZE = 16
IMG_W = 320
IMG_H = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHARS = "0123456789ABCDEFGHKLMNPRSTUVXYZĐ" # Q W I O J 

print("="*60)
print(f"🖥️  Thiết bị đánh giá: {DEVICE.upper()}")
print(f"📊 Đường dẫn tập dữ liệu TEST: {TEST_DIR}")
print(f"🧠 Trọng số mô hình tải vào: {MODEL_PATH}")
print("="*60)

# =====================================================
# DATASET ĐẦU VÀO TẬP TEST (Ảnh gốc không cắt đôi)
# =====================================================
class PARSeqTestDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        if not os.path.exists(root_dir):
            print(f"❌ Không tìm thấy thư mục TEST: {root_dir}")
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
            # Chỉ áp dụng tiền xử lý ảnh xám gốc của hệ thống, không cắt đôi biển
            processed_gray = preprocess_plate(image) 
            
            if len(processed_gray.shape) == 2:
                processed_rgb = cv2.cvtColor(processed_gray, cv2.COLOR_GRAY2RGB)
            else:
                processed_rgb = cv2.cvtColor(processed_gray, cv2.COLOR_BGR2RGB)

        image_tensor = self.transform(processed_rgb)
        return image_tensor, text

def collate_fn(batch):
    images, texts = zip(*batch)
    return torch.stack(images), list(texts)

# =====================================================
# MAIN RUN
# =====================================================
if __name__ == "__main__":
    test_dataset = PARSeqTestDataset(TEST_DIR)
    if len(test_dataset) == 0:
        print("❌ Không có dữ liệu để đánh giá. Vui lòng kiểm tra lại đường dẫn TEST_DIR.")
        sys.exit()

    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True, collate_fn=collate_fn
    )

    model = PARSeq(
        charset_train=CHARS, charset_test=CHARS, max_label_length=12, batch_size=BATCH_SIZE,       
        lr=0.0005, warmup_pct=0.07, weight_decay=1e-4, img_size=[IMG_H, IMG_W],     
        patch_size=[16, 16], embed_dim=384, enc_num_heads=12, enc_mlp_ratio=4,           
        enc_depth=12, dec_num_heads=12, dec_mlp_ratio=4, dec_depth=1,                 
        perm_num=6, perm_forward=True, perm_mirrored=True, decode_ar=True,              
        refine_iters=1, dropout=0.1                  
    )

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Không tìm thấy file trọng số tại: {MODEL_PATH}")
        sys.exit()
        
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model = model.to(DEVICE).eval()

    total_samples = 0
    correct_plates = 0          
    total_char_errors = 0       
    total_gt_chars = 0          
    length_stats = {} 
    wrong_cases_log = []

    print("\n🚀 PARSeq đang tiến hành quét tập dữ liệu TEST...")
    with torch.no_grad():
        for images, texts in tqdm(test_loader, desc="Testing"):
            images = images.to(DEVICE)
            preds = model.forward(images)
            decode_output = model.tokenizer.decode(preds)
            
            decoded_strings = decode_output[0] if isinstance(decode_output, tuple) else decode_output

            for i in range(len(texts)):
                gt = texts[i]
                raw_pred = str(decoded_strings[i])
                pred = raw_pred.split('[E]')[0].strip()
                
                char_dist = editdistance.eval(gt, pred)
                is_perfect = (gt == pred)
                
                total_samples += 1
                total_char_errors += char_dist
                total_gt_chars += len(gt)
                
                if is_perfect:
                    correct_plates += 1
                else:
                    # Lưu lại trường hợp sai để in ra terminal
                    wrong_cases_log.append({
                        "Ground_Truth": gt,
                        "PARSeq_Prediction": pred,
                        "Error_Count": char_dist
                    })
                
                gt_len = len(gt)
                if gt_len not in length_stats:
                    length_stats[gt_len] = {"total": 0, "correct": 0}
                length_stats[gt_len]["total"] += 1
                if is_perfect:
                    length_stats[gt_len]["correct"] += 1

    # --- IN CÁC TRƯỜNG HỢP ĐỌC SƠ SAI RA TERMINAL ---
    print("\n" + "!"*75)
    print(f"🚨 CHI TIẾT TẤT CẢ CÁC BIỂN SỐ BỊ ĐỌC SƠ SAI TRÊN TẬP TEST ({len(wrong_cases_log)}/{total_samples} mẫu):")
    print("!"*75)
    if len(wrong_cases_log) == 0:
        print(" 🎉 Tuyệt vời! Mô hình đạt độ chính xác 100%, không có biển nào bị đọc sai.")
    else:
        print(f"  {'STT':<5} | {'Nhãn Gốc (Ground Truth)':<22} | {'PARSeq Đoán SAI':<22} | {'Lỗi (Ký tự)'}")
        print("  " + "-"*70)
        for idx, case in enumerate(wrong_cases_log):
            print(f"  {idx+1:<5} | {case['Ground_Truth']:<22} | {case['PARSeq_Prediction']:<22} | {case['Error_Count']} ký tự")
    print("!"*75)

    # --- IN BÁO CÁO METRICS TỔNG QUAN ---
    plate_accuracy = (correct_plates / total_samples) * 100
    wer = ((total_samples - correct_plates) / total_samples) * 100
    cer = (total_char_errors / total_gt_chars) * 100

    print("\n" + "="*65)
    print("📊 ĐÁNH GIÁ TỔNG QUAN PARSEQ trên tập TEST:")
    print("="*65)
    print(f"  - Tổng số mẫu biển số thử nghiệm: {total_samples}")
    print(f"  - Số biển nhận diện ĐÚNG HOÀN TOÀN: {correct_plates} / {total_samples}")
    print("-" * 65)
    print(f"  - 🎯 ĐỘ CHÍNH XÁC TOÀN BIỂN (Plate Accuracy) : {plate_accuracy:.2f}%")
    print(f"  - 📉 Tỷ lệ lỗi cấp độ từ/biển số (WER)       : {wer:.2f}%")
    print(f"  - 📉 Tỷ lệ lỗi cấp độ ký tự đơn lẻ (CER)     : {cer:.2f}%")
    print("="*65)

    # 📥 XỬ LÝ 1: LƯU KẾT QUẢ VÀO FILE CSV
    parseq_data = {
        "Metric": ["Plate Accuracy", "Word Error Rate (WER)", "Character Error Rate (CER)"],
        "Value_Percent": [round(plate_accuracy, 2), round(wer, 2), round(cer, 2)]
    }
    df_parseq = pd.DataFrame(parseq_data)
    csv_path = "./model/parseq_eval_1.csv"
    df_parseq.to_csv(csv_path, index=False)
    print(f"💾 Đã lưu bảng số liệu test PARSeq vào: {csv_path}")

    # 📊 XỬ LÝ 2: VẼ ĐỒ THỊ VÀ XUẤT FILE PNG
    plt.figure(figsize=(8, 5))
    colors = ['#2ca02c', '#d62728', '#ff7f0e']
    bars = plt.bar(df_parseq["Metric"], df_parseq["Value_Percent"], color=colors, width=0.4, edgecolor='black')
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1.5, f'{height}%', ha='center', va='bottom', fontweight='bold')

    plt.title("PARSeq Transformer Evaluation Metrics (Test Dataset)", fontsize=12, fontweight='bold', pad=15)
    plt.ylabel("Tỷ lệ phần trăm (%)", fontsize=10)
    plt.ylim(0, max(110, max(df_parseq["Value_Percent"]) + 10))
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    png_path = "./model/parseq_eval_1.png"
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"🖼️  Đã xuất đồ thị trực quan báo cáo PARSeq: {png_path}\n")