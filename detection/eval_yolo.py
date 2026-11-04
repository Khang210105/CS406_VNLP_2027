"""
eval_yolo.py
========================================================================
Đánh giá chất lượng mô hình YOLOv11s phát hiện biển số xe trên tập TEST.
In ra các chỉ số Precision, Recall, mAP và thống kê các loại Loss.
========================================================================
"""
from ultralytics import YOLO
import os
import pandas as pd
import matplotlib.pyplot as plt

def evaluate_model():
    # 1. Đường dẫn tới file trọng số tốt nhất vừa train xong
    model_path = r"runs/detect/bien_so_detector-2/weights/best.pt"
    
    # Kiểm tra xem file weight có tồn tại thật không
    if not os.path.exists(model_path):
        print(f"❌ Không tìm thấy file trọng số tại: {model_path}")
        print("Vui lòng kiểm tra lại chính xác tên folder bạn vừa train nhé!")
        return

    print("🚀 ĐANG NẠP TRỌNG SỐ VÀ TIẾN HÀNH ĐÁNH GIÁ TRÊN TẬP TEST...")
    model = YOLO(model_path)

    # 2. Chạy tính toán số liệu trên tập test bằng hàm .val()
    metrics = model.val(
        data="./train_data/detect_data/data.yaml",
        split="test",          # Ép buộc YOLO chạy trên tập 'test' thay vì 'val'
        imgsz=640,
        batch=4,               
        device=0,
        verbose=True           
    )

    p_val = metrics.results_dict['metrics/precision(B)'] * 100
    r_val = metrics.results_dict['metrics/recall(B)'] * 100
    map50_val = metrics.results_dict['metrics/mAP50(B)'] * 100
    map50_95_val = metrics.results_dict['metrics/mAP50-95(B)'] * 100

    print("\n" + "="*60)
    print("📊 KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH (DETECTION METRICS):")
    print("="*60)
    print(f"🔹 Precision (P)      : {p_val:.2f}%")
    print(f"🔹 Recall (R)         : {r_val:.2f}%")
    print(f"🌟 mAP50              : {map50_val:.2f}%")
    print(f"🌟 mAP50-95           : {map50_95_val:.2f}%")
    print("-" * 60)

    # 📥 XỬ LÝ 1: LƯU KẾT QUẢ VÀO FILE CSV
    data_dict = {
        "Metric": ["Precision", "Recall", "mAP50", "mAP50-95"],
        "Value_Percent": [round(p_val, 2), round(r_val, 2), round(map50_val, 2), round(map50_95_val, 2)]
    }
    df = pd.DataFrame(data_dict)
    csv_path = "yolo_test_metrics_2.csv"
    df.to_csv(csv_path, index=False)
    print(f"💾 Đã lưu bảng số liệu test vào: {csv_path}")

    # 📊 XỬ LÝ 2: VẼ ĐỒ THỊ VÀ XUẤT FILE PNG
    plt.figure(figsize=(8, 5))
    colors = ['#1f77b4', '#aec7e8', '#2ca02c', '#98df8a']
    bars = plt.bar(df["Metric"], df["Value_Percent"], color=colors, width=0.5, edgecolor='black')
    
    # Ghi số liệu lên đầu mỗi cột
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1.5, f'{height}%', ha='center', va='bottom', fontweight='bold')

    plt.title("YOLOv11s_2 Bounding Box Evaluation Metrics (Test Dataset)", fontsize=12, fontweight='bold', pad=15)
    plt.ylabel("Độ chính xác (%)", fontsize=10)
    plt.ylim(0, 110)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    png_path = "yolo_test_metrics_2.png"
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"🖼️  Đã xuất đồ thị trực quan: {png_path}\n")

if __name__ == '__main__':
    evaluate_model()