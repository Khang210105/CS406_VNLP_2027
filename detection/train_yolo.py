"""
train_yolo.py
====================
Huấn luyện YOLOv11s để phát hiện vùng biển số xe trong ảnh.
Chạy sau yolo_split.py.
"""

from ultralytics import YOLO


def main():
    # Đổi từ yolo11n → yolo11s: params tăng ~3.6 lần, mAP tăng ~7.5%
    model = YOLO("yolo11s.pt")

    print("🚀 BẮT ĐẦU HUẤN LUYỆN MODEL PHÁT HIỆN BIỂN SỐ...")

    # model.train(
    #     data="./train_data/detect_data/data.yaml",
    #     epochs=150,           # tăng từ 100 → 150, loss chưa plateau ở epoch 50
    #     imgsz=640,
    #     batch=4,
    #     device=0,
    #     workers=2,
    #     cache=False,
    #     name="bien_so_detector",
    #     exist_ok=False,

    #     # ── Optimizer ──────────────────────────────────────────
    #     optimizer="AdamW",
    #     lr0=0.0005,           # giảm từ 0.001 → 0.0005, ổn định hơn
    #     lrf=0.01,             # lr cuối = lr0 * lrf = 0.000005
    #     warmup_epochs=3,      # 3 epoch đầu lr tăng dần, tránh loss đột biến
    #     cos_lr=True,          # cosine annealing thay vì linear decay
    #     weight_decay=0.0005,

    #     # ── Augmentation ───────────────────────────────────────
    #     # Màu sắc & sáng (biển số ngoài trời, nhiều điều kiện)
    #     hsv_h=0.015,
    #     hsv_s=0.5,
    #     hsv_v=0.4,            # quan trọng: ban ngày/đêm, ngược sáng

    #     # Góc nghiêng (xe đỗ không thẳng hoàn toàn)
    #     degrees=10,
    #     shear=2.0,

    #     # Scale (camera xa/gần khác nhau)
    #     scale=0.5,

    #     # Zoom in: mô phỏng ảnh chụp cận cảnh hơn
    #     # kết hợp với data Roboflow Zoom In augmentation
    #     mosaic=1.0,           # ghép 4 ảnh, giúp detect biển số nhỏ

    #     # QUAN TRỌNG: tắt flip vì biển số bị gương là sai hoàn toàn
    #     fliplr=0.0,
    #     flipud=0.0,
    # )

    model.train(
        data="./train_data/detect_data/data.yaml",
        epochs=200,                  # Tăng lên 300 để loss kịp plateau
        imgsz=640,
        batch=4,                     # Tăng lên 8 hoặc 16 nếu VRAM cho phép (giúp giảm răng cưa Val Loss)
        device=0,
        workers=4,
        cache=False,
        name="bien_so_detector",
        exist_ok=False,

        # ── Optimizer & LR ─────────────────────────────────────
        optimizer="SGD",             # Thử nghiệm SGD giúp mAP mượt và cao hơn ở giai đoạn sau
        lr0=0.01,                    # LR mặc định của SGD thường là 0.01
        lrf=0.01,                    
        warmup_epochs=5,             # Tăng warmup lên 5 chu kỳ
        cos_lr=True,                 
        weight_decay=0.0005,

        # ── Augmentation ───────────────────────────────────────
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.4,            

        degrees=10,
        shear=2.0,
        scale=0.3,                   # Giảm scale một chút để tránh mất chi tiết biển số nhỏ
        
        mosaic=1.0,           
        close_mosaic=20,             # Tắt mosaic ở 20 epoch cuối để mô hình hội tụ trên ảnh thực tế

        # Tuyệt đối KHÔNG lật ảnh biển số
        fliplr=0.0,
        flipud=0.0,
    )
    print("✅ HUẤN LUYỆN HOÀN TẤT!")
    print("📁 Model lưu tại: runs/detect/bien_so_detector/weights/best.pt")


if __name__ == "__main__":
    # Bắt buộc có guard này khi chạy YOLO train trên Windows
    main()