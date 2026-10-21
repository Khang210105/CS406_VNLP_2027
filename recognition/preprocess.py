"""
preprocess.py
=============
Module tiền xử lý ảnh biển số: Tăng độ phân giải -> Gray -> Deskew (xoay thẳng) 
-> Enhance (giảm nhiễu, làm nét) -> Merge (nối dòng) -> Resize chuẩn.
"""

import cv2
import numpy as np

IMG_W = 320
IMG_H = 128

def increase_resolution(image: np.ndarray,
                         min_w: int = IMG_W,
                         min_h: int = IMG_H) -> np.ndarray:
    """
    Upscale thích nghi: chỉ tăng khi ảnh nhỏ hơn kích thước chuẩn.
    Tránh upscale lãng phí với ảnh đã đủ lớn.
    """
    h, w = image.shape[:2]

    # Tính scale cần thiết để đạt ít nhất min_w × min_h
    scale_w = min_w / w
    scale_h = min_h / h
    scale   = max(scale_w, scale_h)

    # Không cần upscale nếu ảnh đã đủ lớn
    if scale <= 1.0:
        return image

    # Làm tròn lên bội số nguyên gần nhất cho gọn
    scale = max(2, round(scale))

    new_w = w * scale
    new_h = h * scale

    return cv2.resize(image, (new_w, new_h),
                      interpolation=cv2.INTER_LANCZOS4)

def to_gray(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def four_point_transform(image: np.ndarray) -> np.ndarray:
    """
    Tìm contour biển số → lấy 4 góc → perspective transform về hình chữ nhật.
    Phù hợp khi ảnh crop từ YOLO vẫn còn bị xéo.
    """
    gray = to_gray(image) if len(image.shape) == 3 else image
    
    # Threshold để tìm vùng sáng (biển số thường sáng hơn nền)
    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Tìm contour lớn nhất (thường là viền biển số)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
    
    c = max(contours, key=cv2.contourArea)
    
    # Xấp xỉ contour thành tứ giác
    peri   = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    
    if len(approx) != 4:
        return image  # Không tìm được 4 góc → giữ nguyên
    
    # Sắp xếp 4 góc: top-left, top-right, bottom-right, bottom-left
    pts = approx.reshape(4, 2).astype(np.float32)
    rect = _order_points(pts)
    
    # Perspective transform về (IMG_W x IMG_H)
    dst = np.array([
        [0,         0        ],
        [IMG_W - 1, 0        ],
        [IMG_W - 1, IMG_H - 1],
        [0,         IMG_H - 1]
    ], dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (IMG_W, IMG_H),
                                 flags=cv2.INTER_CUBIC)
    return warped


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Sắp xếp 4 điểm theo thứ tự: TL, TR, BR, BL."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]     # top-left: x+y nhỏ nhất
    rect[2] = pts[np.argmax(s)]     # bottom-right: x+y lớn nhất
    rect[1] = pts[np.argmin(diff)]  # top-right: x-y nhỏ nhất
    rect[3] = pts[np.argmax(diff)]  # bottom-left: x-y lớn nhất
    return rect

def deskew(gray: np.ndarray) -> np.ndarray:
    """
    Cải tiến: kết hợp HoughLines + MinAreaRect để ước lượng góc chính xác hơn.
    """
    h, w = gray.shape[:2]

    # ── Phương pháp 1: HoughLinesP (như cũ nhưng tham số tốt hơn) ──
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=60,           # tăng từ 50 → ít bắt nhiễu hơn
        minLineLength=w * 0.2,  # ít nhất 20% chiều rộng ảnh → bỏ đường ngắn
        maxLineGap=20
    )

    hough_angle = None
    if lines is not None:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) < 1e-5:
                continue
            angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
            if abs(angle) < 30:
                angles.append(angle)
        if angles:
            hough_angle = np.median(angles)

    # ── Phương pháp 2: MinAreaRect (backup nếu HoughLines thất bại) ──
    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    rect_angle = None
    if len(coords) >= 5:
        angle = cv2.minAreaRect(coords)[-1]
        # Chuẩn hóa về [-45, 45]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) < 30:
            rect_angle = angle

    # ── Chọn góc: ưu tiên HoughLines, fallback MinAreaRect ──
    final_angle = None
    if hough_angle is not None:
        final_angle = hough_angle
    elif rect_angle is not None:
        final_angle = rect_angle

    # Bỏ qua nếu góc quá nhỏ (< 1 độ) → không cần xoay
    if final_angle is None or abs(final_angle) < 1.0:
        return gray

    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, final_angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)

def enhance_image(gray: np.ndarray) -> np.ndarray:

    # 1. Denoise nhẹ hơn — dùng bilateral thay NlMeans
    # Bilateral giữ cạnh sắc, chỉ làm mờ vùng phẳng (nền)
    # Nhanh hơn NlMeans ~5-10 lần
    gray = cv2.bilateralFilter(gray, d=5, sigmaColor=30, sigmaSpace=30)

    # 2. CLAHE — giữ nguyên, đang tốt
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray  = clahe.apply(gray)

    # 3. Sharpen thích nghi — chỉ sharpen những vùng có cạnh
    # Laplacian đo độ sắc nét cục bộ, vùng nào đã sắc thì không sharpen thêm
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = laplacian.var()   # đo độ sắc tổng thể của ảnh

    if sharpness < 200:           # ảnh mờ → sharpen mạnh
        alpha = 1.6
    elif sharpness < 500:         # ảnh trung bình → sharpen nhẹ
        alpha = 1.3
    else:                         # ảnh đã sắc → không sharpen
        alpha = 1.0

    if alpha > 1.0:
        blurred   = cv2.GaussianBlur(gray, (0, 0), sigmaX=2)
        gray      = cv2.addWeighted(gray, alpha, blurred, 1.0 - alpha, 0)

    return gray

def preprocess_plate(image: np.ndarray) -> np.ndarray:

    # 1. Perspective transform (xử lý ảnh xéo trước)
    image = four_point_transform(image)

    # 2. Tăng độ phân giải
    img_upscaled = increase_resolution(image, min_w=IMG_W, min_h=IMG_H)

    # 3. Chuyển xám
    gray = to_gray(img_upscaled)

    # 4. Deskew cải tiến (xử lý nghiêng nhẹ còn lại sau perspective)
    gray = deskew(gray)

    # 5. Enhance
    gray = enhance_image(gray)

    # 6. Resize chuẩn
    gray = cv2.resize(gray, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)

    assert gray.ndim == 2, f"[preprocess_plate] Kỳ vọng ảnh xám 2D, nhận được shape={gray.shape}"
    return gray