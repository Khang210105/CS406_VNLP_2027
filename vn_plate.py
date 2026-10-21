"""
vn_plate.py
===========
Post-processing chuẩn xác cho biển số xe máy Việt Nam.
Phân biệt rõ ràng: Biển máy điện (41MĐ1-12345) và xe máy thường (59AA-12345, 59A3-12345, 51NG-16653).
"""

import re

# =====================================================
# PROVINCE MAP
# =====================================================
PROVINCE_MAP = {
    "11": "Cao Bằng", "12": "Lạng Sơn", "14": "Quảng Ninh",
    "15": "Hải Phòng", "16": "Hải Phòng", "17": "Thái Bình",
    "18": "Nam Định", "19": "Phú Thọ", "20": "Thái Nguyên",
    "21": "Yên Bái", "22": "Tuyên Quang", "23": "Hà Giang",
    "24": "Lào Cai", "25": "Lai Châu", "26": "Sơn La",
    "27": "Điện Biên", "28": "Hòa Bình", "29": "Hà Nội",
    "30": "Hà Nội", "31": "Hà Nội", "32": "Hà Nội",
    "33": "Hà Nội", "34": "Hải Dương", "35": "Ninh Bình",
    "36": "Thanh Hóa", "37": "Nghệ An", "38": "Hà Tĩnh",
    "39": "Đồng Nai", "40": "Hà Nội", "41": "TP. Hồ Chí Minh",
    "43": "Đà Nẵng", "47": "Đắk Lắk", "48": "Đắk Nông",
    "49": "Lâm Đồng", "50": "TP. Hồ Chí Minh", "51": "TP. Hồ Chí Minh",
    "52": "TP. Hồ Chí Minh", "53": "TP. Hồ Chí Minh", "54": "TP. Hồ Chí Minh",
    "55": "TP. Hồ Chí Minh", "56": "TP. Hồ Chí Minh", "57": "TP. Hồ Chí Minh",
    "58": "TP. Hồ Chí Minh", "59": "TP. Hồ Chí Minh", "60": "Đồng Nai",
    "61": "Bình Dương", "62": "Long An", "63": "Tiền Giang",
    "64": "Vĩnh Long", "65": "Cần Thơ", "66": "Đồng Tháp",
    "67": "An Giang", "68": "Kiên Giang", "69": "Cà Mau",
    "70": "Tây Ninh", "71": "Bến Tre", "72": "Bà Rịa - Vũng Tàu",
    "73": "Quảng Bình", "74": "Quảng Trị", "75": "Thừa Thiên Huế",
    "76": "Quảng Ngãi", "77": "Bình Định", "78": "Phú Yên",
    "79": "Khánh Hòa", "80": "Cơ quan Trung ương", "81": "Gia Lai",
    "82": "Kon Tum", "83": "Sóc Trăng", "84": "Trà Vinh",
    "85": "Ninh Thuận", "86": "Bình Thuận", "88": "Vĩnh Phúc",
    "89": "Hưng Yên", "90": "Hà Nam", "92": "Quảng Nam",
    "93": "Bình Phước", "94": "Bạc Liêu", "95": "Hậu Giang",
    "97": "Bắc Kạn", "98": "Bắc Giang", "99": "Bắc Ninh"
}

# Bảng hoán đổi chữ và số khi bị nhận diện nhầm
CHAR_TO_DIGIT = {
    'O':'0', 'Q':'0', 'D':'0', 'I':'1', 'Z':'2', 'S':'5', 'G':'6', 'B':'8', 'A':'4'
}

DIGIT_TO_CHAR = {
    '0': 'D', '8': 'B', '5': 'S', '6': 'G', '4': 'A'
}

def smart_fix_series(chars):
    total_len = len(chars)
    if total_len < 4:
        return chars

    # 1. KIỂM TRA BIỂN NGOẠI GIAO (NG) TRƯỚC
    if chars[2] == 'N' and chars[3] in ['6', '0', 'G']:
        chars[3] = 'G'
        for i in range(4, total_len):
            if chars[i] in CHAR_TO_DIGIT:
                chars[i] = CHAR_TO_DIGIT[chars[i]]
        return chars

    # 2. XỬ LÝ THEO TỪNG LOẠI BIỂN DỰA TRÊN CHIỀU DÀI CHUỖI LÀM SẠCH

    # Trường hợp: BIỂN XE MÁY ĐIỆN (10 ký tự, Ví dụ: 41MĐ112345)
    if total_len == 10:
        # Ký tự thứ 3 (Index 2) phải là CHỮ (Thường là M)
        if chars[2] in DIGIT_TO_CHAR:
            chars[2] = DIGIT_TO_CHAR[chars[2]]
        
        # Ký tự thứ 4 (Index 3) phải là chữ Đ. Sửa lỗi OCR nhầm thành D hoặc 0
        if chars[3] in ['D', '0', 'Q', 'O']:
            chars[3] = 'Đ'
        
        # Ký tự thứ 5 (Index 4) BẮT BUỘC PHẢI LÀ SỐ (Số thứ tự phân loại xe điện, ví dụ: 1)
        if chars[4] in CHAR_TO_DIGIT:
            chars[4] = CHAR_TO_DIGIT[chars[4]]
            
        # Các ký tự còn lại từ Index 5 đến hết bắt buộc là số
        for i in range(5, 10):
            if chars[i] in CHAR_TO_DIGIT:
                chars[i] = CHAR_TO_DIGIT[chars[i]]

    # Trường hợp: BIỂN XE MÁY THƯỜNG 5 SỐ (9 ký tự, Ví dụ: 59AA12345 hoặc 59A312345)
    elif total_len == 9:
        # Ký tự thứ 3 (Index 2) bắt buộc phải là CHỮ (Chữ cái đầu của seri, ví dụ: A)
        if chars[2] in DIGIT_TO_CHAR:
            chars[2] = DIGIT_TO_CHAR[chars[2]]
        
        # Ký tự thứ 4 (Index 3) có thể là CHỮ (AA) hoặc SỐ (A3). 
        # Phần này thả lỏng để tầng Regex của is_valid_plate kiểm tra sẽ chính xác hơn,
        # Tránh việc ép sai từ số sang chữ hoặc ngược lại.
        
        # Các ký tự từ Index 4 đến hết bắt buộc phải là số (5 số dòng dưới)
        for i in range(4, 9):
            if chars[i] in CHAR_TO_DIGIT:
                chars[i] = CHAR_TO_DIGIT[chars[i]]

    # Trường hợp: BIỂN 4 SỐ ĐỜI CŨ (8 ký tự, Ví dụ: 59T12345)
    elif total_len == 8:
        if chars[2] in DIGIT_TO_CHAR:
            chars[2] = DIGIT_TO_CHAR[chars[2]]
        # Ký tự thứ 4 (Index 3) bắt buộc phải là SỐ (Ví dụ số 1 trong T1)
        if chars[3] in CHAR_TO_DIGIT:
            chars[3] = CHAR_TO_DIGIT[chars[3]]

    return chars

# =====================================================
# VALIDATION VIA REGEX
# =====================================================
def is_valid_plate(text: str) -> bool:
    province = text[:2]
    if province not in PROVINCE_MAP:
        return False

    patterns = [
        # 1. Biển máy điện 10 ký tự: 2 số tỉnh + 2 chữ cái (có Đ) + 6 số (Ví dụ: 41MĐ112345)
        r"^\d{2}[A-Z]Đ\d{6}$",

        # 2. Biển Ngoại Giao: 2 số tỉnh + NG + 5 số (Ví dụ: 51NG16653)
        r"^\d{2}NG\d{5}$",

        # 3. Biển 5 số xe máy thường loại mới: 2 số tỉnh + 2 chữ cái + 5 số (Ví dụ: 59AA12345)
        r"^\d{2}[A-Z]{2}\d{5}$",

        # 4. Biển 5 số xe máy thường loại cũ: 2 số tỉnh + 1 chữ cái + 1 số + 5 số (Ví dụ: 59A312345)
        r"^\d{2}[A-Z]\d{6}$",

        # 5. Biển 4 số cũ: 2 số tỉnh + 1 chữ cái + 1 số + 4 số (Ví dụ: 59T1234)
        r"^\d{2}[A-Z]\d{5}$",
        # Biển 4 số cũ: 2 chữ cái (Ví dụ: 68AA1111)
        r"^\d{2}[A-Z]{2}\d{4}$"
    ]

    return any(re.fullmatch(p, text) for p in patterns)

# =====================================================
# POST PROCESS
# =====================================================
def postprocess_ocr(text: str, return_two_lines: bool = True):
    # Làm sạch chuỗi, giữ lại chữ Đ
    text = text.replace("-", "").replace(".", "").replace(" ", "").upper()

    if len(text) < 4:
        return {"formatted": text, "province": "Không xác định", "valid": False}

    chars = list(text)

    # 2 ký tự đầu luôn là mã tỉnh (Số)
    for i in range(min(2, len(chars))):
        if chars[i] in CHAR_TO_DIGIT:
            chars[i] = CHAR_TO_DIGIT[chars[i]]

    # Chạy bộ lọc sửa lỗi thông minh
    chars = smart_fix_series(chars)

    text = "".join(chars)
    valid = is_valid_plate(text)

    province_code = text[:2]
    province_name = PROVINCE_MAP.get(province_code, "Không xác định")

    if not valid:
        return {"formatted": text, "province": province_name, "valid": False}

    # =====================================================
    # ĐỊNH DẠNG ĐẦU RA HIỂN THỊ (FORMATTER)
    # =====================================================
    province = text[:2]
    
    # 1. Định dạng Biển Ngoại Giao NG
    if text[2:4] == "NG":
        series = "NG"
        tail = text[4:]
        if len(tail) == 5:
            tail = f"{tail[:3]}.{tail[3:]}"
        formatted = f"{province}-{series}\n{tail}" if return_two_lines else f"{province}-{series} {tail}"
        return {"formatted": formatted, "province": province_name, "valid": True}

    # 2. Định dạng Biển Máy Điện 10 ký tự (Dòng trên 5 ký tự: Mã tỉnh + Chữ + Đ + Số)
    if len(text) == 10:
        series = text[2:5]  # Lấy 3 ký tự seri dòng trên (Ví dụ: MĐ1)
        tail = text[5:]     # 5 số dòng dưới
        tail = f"{tail[:3]}.{tail[3:]}"
        formatted = f"{province}-{series}\n{tail}" if return_two_lines else f"{province}-{series} {tail}"
        return {"formatted": formatted, "province": province_name, "valid": True}

    # 3. Định dạng Biển Xe Máy Thường 9 ký tự (Dòng trên 4 ký tự: Mã tỉnh + Seri 2 ký tự)
    series = text[2:4]  # Lấy 2 ký tự seri dòng trên (Ví dụ: AA hoặc A3)
    tail = text[4:]     # Dãy số dòng dưới
    
    if len(tail) == 5:
        tail = f"{tail[:3]}.{tail[3:]}"  # Thêm dấu chấm cho biển 5 số (123.45)

    if return_two_lines:
        formatted = f"{province}-{series}\n{tail}"
    else:
        formatted = f"{province}-{series} {tail}"

    return {"formatted": formatted, "province": province_name, "valid": True}