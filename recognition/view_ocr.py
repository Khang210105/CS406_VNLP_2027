"""
view_ocr.py
========================================================================
Tạo giao diện Web siêu nhẹ để duyệt qua toàn bộ cặp Ảnh + Nhãn trong extra_ocr
========================================================================
"""
import os
from flask import Flask, render_template_string, send_from_directory

app = Flask(__name__)

# Đường dẫn đến thư mục extra_ocr của bạn
DATA_DIR = r"C:\Users\MrBeast\Desktop\CS338_PR\project\ocr_split\test"

@app.route('/file/<path:filename>')
def get_file(filename):
    # Hàm bổ trợ để Flask có thể load được ảnh từ đường dẫn tuyệt đối ngoài hệ thống
    return send_from_directory(DATA_DIR, filename)

@app.route('/')
def index():
    samples_data = []
    
    if not os.path.exists(DATA_DIR):
        return f"<h2>❌ Không tìm thấy thư mục: {DATA_DIR}</h2>"

    # Duyệt qua tất cả các thư mục con (sample_002005, sample_002006,...)
    subfolders = sorted([f for f in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, f))])

    for folder in subfolders:
        folder_path = os.path.join(DATA_DIR, folder)
        
        # Tìm file ảnh và file txt trong thư mục con này
        all_files = os.listdir(folder_path)
        img_name = None
        txt_name = None
        
        for file in all_files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                img_name = file
            elif file.lower().endswith('.txt'):
                txt_name = file

        # Nếu có ảnh thì tiến hành lấy thông tin
        if img_name:
            label_content = "[Trống hoặc không có file .txt]"
            if txt_name:
                txt_path = os.path.join(folder_path, txt_name)
                try:
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        label_content = f.read().strip()
                except Exception:
                    label_content = "[Lỗi đọc file txt]"

            # Lưu đường dẫn tương đối để giao diện Web gọi qua hàm get_file
            img_src = f"/file/{folder}/{img_name}"
            
            samples_data.append({
                "folder": folder,
                "img_src": img_src,
                "label": label_content
            })

    # Giao diện HTML được nhúng trực tiếp bằng mã Python (Không cần tạo file .html riêng)
    html_template = """
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <title>OCR Dataset Viewer</title>
        <style>
            body { font-family: 'Segoe UI', Arial, sans-serif; background-color: #0d1117; color: #e6edf3; padding: 20px; margin: 0; }
            h1 { text-align: center; color: #58a6ff; }
            .stats { text-align: center; color: #8b949e; margin-bottom: 20px; font-size: 16px; }
            .grid-container { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 20px; padding: 10px; }
            .card { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; overflow: hidden; display: flex; flex-direction: column; align-items: center; padding: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
            .card:hover { border-color: #58a6ff; }
            .folder-title { font-size: 12px; color: #8b949e; margin-bottom: 8px; font-weight: bold; }
            .img-box { width: 100%; height: 120px; display: flex; align-items: center; justify-content: center; background-color: #010409; border-radius: 4px; overflow: hidden; }
            .img-box img { max-width: 100%; max-height: 100%; object-fit: contain; }
            .label-box { margin-top: 10px; width: 100%; text-align: center; background-color: #21262d; padding: 6px 0; border-radius: 4px; font-size: 16px; font-weight: bold; color: #ff79c6; border: 1px dashed #30363d; white-space: pre-line; }
        </style>
    </head>
    <body>
        <h1>🗂️ Trình Duyệt Dữ Liệu Bộ Nhãn extra_ocr</h1>
        <div class="stats">Tổng số mẫu dữ liệu phát hiện: <strong>{{ total }}</strong></div>
        
        <div class="grid-container">
            {% for item in samples %}
            <div class="card">
                <div class="folder-title">{{ item.folder }}</div>
                <div class="img-box">
                    <img src="{{ item.img_src }}" alt="OCR Target">
                </div>
                <div class="label-box">{{ item.label }}</div>
            </div>
            {% endfor %}
        </div>
    </body>
    </html>
    """
    return render_template_string(html_template, samples=samples_data, total=len(samples_data))

if __name__ == '__main__':
    # Chạy ứng dụng ở cổng 5050 để không trùng với app chính
    print("🚀 Giao diện đang khởi động... Hãy mở trình duyệt truy cập: http://127.0.0.1:5050")
    app.run(host='0.0.0.0', port=5050, debug=True)