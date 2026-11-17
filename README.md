# CS406 – Nhận diện biển số xe máy Việt Nam

YOLO phát hiện biển → crop → tiền xử lý → OCR (CRNN / PARSeq) → sửa lỗi theo luật biển VN.
Train trên GPU của Kaggle, máy mình không cần GPU.

## 1. Cài đặt (1 lần)

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt  # gồm cả Kaggle CLI (lệnh `kaggle`)
pip install --no-deps "strhub @ git+https://github.com/baudm/parseq.git@1902db043c029a7e03a3818c616c06600af574be"

kaggle auth login                # đăng nhập Kaggle (mở trình duyệt)
```

Trên kaggle.com: **Settings → Phone verification** – bắt buộc, không có thì không dùng được GPU.

## 2. Data

Chọn 1 cách:

* **Dùng data nhóm đã upload**: nhờ người giữ dataset `<owner>/cs406-vnlp-data` share cho bạn trên Kaggle,
  rồi sửa `config.yaml`: `kaggle.data_dataset: "<owner>/cs406-vnlp-data"`.
* **Tự chuẩn bị**: tải [data từ Drive](https://drive.google.com/drive/folders/1AAwdv0Kv0_UM4muJe-nmlt7rWpdf4prR?usp=sharing),
  giải nén **cùng cấp** với thư mục repo:
  ```
  Đồ án/                       ← thư mục cha bất kỳ
  ├── CS406_VNLP_2027/         ← repo này
  └── dataset_CS406/           ← data giải nén ra
      ├── raw/                 ảnh gốc + nhãn YOLO (view1 … view5)
      ├── processed/           crop tự động
      └── processed_manual/    crop đã gán nhãn tay  ← split dùng thư mục này
  ```
  (đặt chỗ khác thì sửa `paths:` trong `config.yaml`), rồi chạy:
  ```bash
  python -m vnlp.data.split_recognition
  python -m vnlp.data.split_detection
  ```

## 3. Train trên Kaggle

```bash
python -m vnlp.kaggle_push quick   # chạy thử vài ảnh (lần đầu nên chạy cái này)
python -m vnlp.kaggle_push         # chạy thật
python -m vnlp.kaggle_push pull    # chạy xong: tải kết quả về outputs/<experiment>/
```

Lệnh in ra link Kaggle để theo dõi (tab **Logs**).

## 4. Thử nghiệm mới

1. Sửa code trong `vnlp/` hoặc tham số trong `config.yaml`.
2. Đổi `experiment:` trong `config.yaml` (vd `baseline` → `parseq-lr1e-4`) để không đè kết quả cũ.
3. `git commit` → `python -m vnlp.kaggle_push` → `pull`.

Mỗi kết quả kèm `config_used.yaml` + `run_info.json` (commit, thời gian) để biết nó từ code/tham số nào.

**Thí nghiệm camera mới (leave-one-domain-out)** – train view3+view4, test toàn bộ camera UIT (view1+view2):

```yaml
# config.yaml
experiment: lodo-uit
split:
  protocol: lodo            # mặc định: stratified
  test_views: [view1, view2]
```

Chạy lại 2 lệnh split (tạo `dataset/splits_lodo-view1-view2/`, không đè split chính) → `python -m vnlp.kaggle_push`.
So sánh với `baseline` ở dòng view1/view2 để biết mô hình tổng quát sang camera chưa thấy tốt đến đâu.

**RQ1 – nguồn dữ liệu nào giúp YOLO tổng quát sang camera chưa thấy** (3 nhánh, cùng 200 ảnh train, cùng test view1+view2+view5):

| Nhánh | `experiment` | `split.train_views` | `split.train_quota` |
|---|---|---|---|
| A – chỉ view3 | `rq1-view3` | `[view3]` | `null` (view3 có đúng 200 ảnh train) |
| B – chỉ view4 | `rq1-view4` | `[view4]` | `{view4: 200}` |
| C – kết hợp | `rq1-mix` | `[view3, view4]` | `{view3: 100, view4: 100}` |

Mỗi nhánh: `protocol: lodo`, `test_views: [view1, view2]`, đặt 2 dòng trên, `notebook.run_detection: true`
(tắt `run_crnn`, `run_parseq` cho nhanh) → `python -m vnlp.data.split_detection` → `python -m vnlp.kaggle_push`.
Split detection mỗi nhánh nằm riêng ở `dataset/splits_lodo-view1-view2_tr-.../` nên không đè nhau.
Cần có sẵn split recognition lodo (`recognition_split_lodo-view1-view2.csv`) vì detection chia đồng bộ theo nó.

---

## Tham khảo

### Chạy local (không qua Kaggle)

```bash
python -m vnlp.recognition.crnn train      # hoặc: eval
python -m vnlp.recognition.parseq train    # hoặc: eval
python -m vnlp.detection.train
python -m vnlp.detection.evaluate
```

Hoặc mở `notebooks/main.ipynb` (kernel `.venv`). `notebook.quick_test: true` để chạy thử trên CPU.

Demo: `python apps/demo/app.py` · Xem nhãn OCR: `python apps/ocr_viewer.py`

### Chuẩn bị data từ đầu

| Bước | Lệnh |
|---|---|
| 1. Crop biển từ ảnh gốc | `python -m vnlp.data.crop_plates` |
| 2. PARSeq gợi ý nhãn | `python -m vnlp.data.label_plates` |
| 3. Review nhãn bằng tay | `crops_dir` → `processed_dir` |
| 4. Chia train/val/test | `split_recognition`, `split_detection` |

* OCR chia theo **biển số** (test toàn biển chưa thấy), đều theo view, 70/15/15.
* Detection chia **cùng split với OCR** → đánh giá end-to-end không rò rỉ.
* view1/2: camera UIT · view3: xe đậu · view4: dataset ngoài · view5: khung rộng của view1 (chỉ detection).

### Cấu trúc

```
config.yaml          mọi cấu hình
vnlp/                code: data/, detection/, recognition/, utils/, pipeline.py, kaggle_push.py
notebooks/main.ipynb notebook chính (Kaggle chạy file này)
apps/                demo web, xem nhãn
dataset/             splits/ + manifest CSV
outputs/<experiment> kết quả
docs/                tài liệu đồ án
```

Data có biển số thật → mọi dataset/notebook trên Kaggle đều **private**.
