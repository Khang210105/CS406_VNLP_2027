import os
import io
import base64
from flask import Flask, render_template, request, jsonify, Response

import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# --- IMPORT MÔ HÌNH PARSEQ ---
from strhub.models.parseq.system import PARSeq

from preprocess import preprocess_plate
from vn_plate import postprocess_ocr

app = Flask(__name__)

# =====================================================================
# ĐƯỜNG DẪN TRỌNG SỐ MÔ HÌNH (Lấy chuẩn từ file app.py của bạn)
# =====================================================================
YOLO_MODEL_PATH   = r"C:\Users\MrBeast\Desktop\CS338_PR\project\runs\detect\bien_so_detector\weights\best.pt"
CRNN_MODEL_PATH   = r"C:\Users\MrBeast\Desktop\CS338_PR\project\model\ocr_model.pth"
PARSEQ_MODEL_PATH = r"C:\Users\MrBeast\Desktop\CS338_PR\project\model\parseq_model.ckpt"

CONF_THRESHOLD  = 0.10
IMG_W      = 320
IMG_H      = 128
CHARS = "0123456789ABCDEFGHKLMNPRSTUVXYZĐ"
idx_to_char = {i + 1: c for i, c in enumerate(CHARS)}
NUM_CLASSES = len(CHARS) + 1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Biến toàn cục dùng để lưu trạng thái Video phục vụ luồng streaming
CURRENT_VIDEO_PATH = None
CURRENT_MODEL_TYPE = "crnn"

# =====================================================================
# ĐỊNH NGHĨA KIẾN TRÚC MÔ HÌNH CRNN
# =====================================================================
class SimpleOCR(nn.Module):
    def __init__(self):
        super(SimpleOCR, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d((2, 1))
        )
        self.rnn = nn.LSTM(input_size=512 * 8, hidden_size=256, num_layers=2, bidirectional=True, dropout=0.3, batch_first=True)
        self.fc = nn.Linear(512, NUM_CLASSES)

    def forward(self, x):
        x = self.cnn(x)
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(b, w, c * h)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x.log_softmax(2)

# Khởi tạo nạp mô hình khi chạy server
print(f"🔄 Loading YOLO detector from: {YOLO_MODEL_PATH}")
yolo_model = YOLO(YOLO_MODEL_PATH)

print(f"🔄 Loading CRNN Model from: {CRNN_MODEL_PATH}")
crnn_model = SimpleOCR().to(DEVICE)
crnn_model.load_state_dict(torch.load(CRNN_MODEL_PATH, map_location=DEVICE, weights_only=True))
crnn_model.eval()

print(f"🔄 Loading PARSeq Transformer from: {PARSEQ_MODEL_PATH}")
parseq_model = PARSeq(
    charset_train=CHARS, charset_test=CHARS, max_label_length=12, batch_size=1,
    lr=0.0005, warmup_pct=0.07, weight_decay=1e-4, img_size=[IMG_H, IMG_W],
    patch_size=[16, 16], embed_dim=384, enc_num_heads=12, enc_mlp_ratio=4,
    enc_depth=12, dec_num_heads=12, dec_mlp_ratio=4, dec_depth=1,
    perm_num=6, perm_forward=True, perm_mirrored=True, decode_ar=True, refine_iters=1, dropout=0.1
)
if os.path.exists(PARSEQ_MODEL_PATH):
    checkpoint = torch.load(PARSEQ_MODEL_PATH, map_location=DEVICE)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        parseq_model.load_state_dict(checkpoint['state_dict'])
    else:
        parseq_model.load_state_dict(checkpoint)
parseq_model = parseq_model.to(DEVICE).eval()

# Khởi tạo Transforms tương ứng từng model
crnn_transform = transforms.ToTensor()
parseq_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# =====================================================================
# CÁC HÀM XỬ LÝ NHẬN DIỆN CHỮ (OCR)
# =====================================================================
def ctc_decode(output: torch.Tensor) -> str:
    pred = output.argmax(2)[0].cpu().tolist()
    text, prev = "", -1
    for p in pred:
        if p != prev and p != 0:
            text += idx_to_char.get(p, "")
        prev = p
    return text

def single_ocr_predict(crop_img, model_type="crnn"):
    gray = preprocess_plate(crop_img)
    if gray is None or gray.size == 0:
        return ""

    if model_type == "parseq":
        if len(gray.shape) == 2:
            processed_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        else:
            processed_rgb = cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)
        x = parseq_transform(processed_rgb).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = parseq_model(x)
            decode_output = parseq_model.tokenizer.decode(out)
            decoded_strings = decode_output[0] if isinstance(decode_output, tuple) else decode_output
            raw_text = str(decoded_strings[0]).split('[E]')[0].strip()
    else:
        x = crnn_transform(Image.fromarray(gray)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = crnn_model(x)
        raw_text = ctc_decode(out)

    try:
        result = postprocess_ocr(raw_text, return_two_lines=True)
        if isinstance(result, dict):
            # Trả về dict chứa cả biển số và tỉnh thành
            return {
                "text": result.get("formatted", raw_text),
                "province": result.get("province", "Không xác định")
            }
        return {"text": result, "province": "Không xác định"}
    except Exception:
        return {"text": raw_text, "province": "Không xác định"}

# Pipeline xử lý ảnh tĩnh gốc của bạn
def process_image_pipeline(image, model_type="crnn"):
    results = yolo_model(image, conf=CONF_THRESHOLD, verbose=False)
    plates_found = []
    
    if len(results) == 0 or results[0].boxes is None:
        return plates_found

    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        cropped = image[y1:y2, x1:x2]
        if cropped.size == 0:
            continue

        # --- TIỀN XỬ LÝ ẢNH ĐỂ LẤY FRAME HIỂN THỊ LÊN FRONTEND ---
        gray = preprocess_plate(cropped)
        
        # Chuyển đổi ảnh cắt (cropped) sang chuỗi Base64
        _, buf_crop = cv2.imencode('.jpg', cropped)
        cropped_b64 = base64.b64encode(buf_crop.tobytes()).decode('utf-8')

        # Chuyển đổi ảnh xám tiền xử lý (gray) sang chuỗi Base64
        if gray is not None and gray.size > 0:
            _, buf_gray = cv2.imencode('.jpg', gray)
            preprocessed_b64 = base64.b64encode(buf_gray.tobytes()).decode('utf-8')
        else:
            preprocessed_b64 = cropped_b64

        # Gọi hàm dự đoán OCR chữ
        ocr_res = single_ocr_predict(cropped, model_type)
        
        plates_found.append({
            "box": [int(x1), int(y1), int(x2), int(y2)],
            "text": ocr_res["text"],          
            "province": ocr_res["province"],
            "cropped_img": cropped_b64,        # Đã bổ sung gửi lên Frontend
            "preprocessed_img": preprocessed_b64 # Đã bổ sung gửi lên Frontend
        })
    return plates_found

def generate_result_image(image, plates):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    for idx, p in enumerate(plates):
        x1, y1, x2, y2 = p["box"]
        rect = mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='#00ff66', linewidth=2.5)
        ax.add_patch(rect)
        
        display_text = p["text"].replace('\n', ' ')
        ax.text(x1, max(y1 - 12, 15), f"[{idx+1}] {display_text}", 
                color='black', fontsize=9, fontweight='bold',
                bbox=dict(facecolor='#00ff66', edgecolor='none', boxstyle='round,pad=0.2'))

    ax.axis('off')
    for spine in ax.spines.values():
        spine.set_visible(True)

    plt.suptitle("License Plate Recognition System", fontsize=14, fontweight="bold", color="#e6edf3", y=1.01)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg', dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return img_base64

# =====================================================================
# RE-STREAMING GENERATOR CHO FILE VIDEO
# =====================================================================
def generate_video_stream():
    global CURRENT_VIDEO_PATH, CURRENT_MODEL_TYPE
    if not CURRENT_VIDEO_PATH or not os.path.exists(CURRENT_VIDEO_PATH):
        return

    cap = cv2.VideoCapture(CURRENT_VIDEO_PATH)
    track_history = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model.track(frame, persist=True, conf=CONF_THRESHOLD, verbose=False, tracker="bytetrack.yaml")
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                cropped = frame[y1:y2, x1:x2]
                if cropped.size == 0:
                    continue

                if track_id not in track_history or track_history[track_id] == "Scanning...":
                    pred_text = single_ocr_predict(cropped, CURRENT_MODEL_TYPE)
                    if pred_text and len(pred_text) >= 4:
                        track_history[track_id] = pred_text
                    else:
                        track_history[track_id] = "Scanning..."

                stable_text = track_history[track_id]
                color = (0, 255, 102) if stable_text != "Scanning..." else (0, 153, 255)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                clean_display = stable_text.replace('\n', ' ')
                label = f"ID:{track_id} | {clean_display}"
                
                cv2.rectangle(frame, (x1, max(y1 - 22, 0)), (x1 + len(label)*9, max(y1, 22)), color, -1)
                cv2.putText(frame, label, (x1, max(y1 - 6, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()

# =====================================================================
# ROUTERS - API ENDPOINTS
# =====================================================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global CURRENT_VIDEO_PATH, CURRENT_MODEL_TYPE
    if 'image' not in request.files:
        return jsonify({"error": "Không tìm thấy file ảnh hoặc video"}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "File trống"}), 400

    model_type = request.form.get("model_type", "crnn")
    filename = file.filename.lower()

    # Kiểm tra xem file tải lên là video hay ảnh dựa trên phần mở rộng đuôi file
    if filename.endswith(('.mp4', '.mov', '.avi', '.mkv')):
        CURRENT_MODEL_TYPE = model_type
        temp_path = os.path.join(os.getcwd(), "temp_input.mp4")
        file.save(temp_path)
        CURRENT_VIDEO_PATH = temp_path
        return jsonify({"is_video": True})
    
    # Nếu là ảnh tĩnh, chạy logic xử lý ảnh mặc định cũ của bạn
    try:
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({"error": "Định dạng file không hợp lệ"}), 400

        plates = process_image_pipeline(image, model_type=model_type)
        result_image_b64 = generate_result_image(image, plates)

        formatted_plates = []
        for p in plates:
            formatted_plates.append({
                "text": p["text"],                         # Đổi từ "plate" thành "text" cho khớp index.html
                "province": p["province"],                 # Đảm bảo tỉnh thành
                "cropped_img": p["cropped_img"],           # Truyền ảnh cắt Base64
                "preprocessed_img": p["preprocessed_img"]   # Truyền ảnh xử lý Base64
            })

        return jsonify({
            "is_video": False,
            "success": True,
            "plates": formatted_plates,
            "image": result_image_b64
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)