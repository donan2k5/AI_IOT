from flask import Flask, request
import os
import random
import cv2
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = r'D:\Study\Python\Iotfpt\image\ANH\Anh_PI'
CROPPED_FOLDER = r'D:\Study\Python\Iotfpt\image\ANH\Anh_PI_Opencv'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CROPPED_FOLDER, exist_ok=True)

def crop_largest_red_or_blue_region(image_path, padding=20):
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Không đọc được ảnh tại: {image_path}")
        return None

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Màu xanh
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Màu đỏ (chia làm 2 khoảng do red nằm ở đầu và cuối của dải HSV)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # Tổng hợp mask
    mask_combined = cv2.bitwise_or(mask_blue, mask_red)

    contours, _ = cv2.findContours(mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("❌ Không tìm thấy vùng màu xanh hoặc đỏ.")
        return None

    largest_cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_cnt)

    # Zoom ra thêm 20 pixel mỗi phía (nếu nằm trong giới hạn ảnh)
    x1 = max(x - padding, 0)
    y1 = max(y - padding, 0)
    x2 = min(x + w + padding, img.shape[1])
    y2 = min(y + h + padding, img.shape[0])

    cropped = img[y1:y2, x1:x2]

    cropped_filename = 'cropped_' + os.path.basename(image_path)
    cropped_path = os.path.join(CROPPED_FOLDER, cropped_filename)
    cv2.imwrite(cropped_path, cropped)
    print(f"✅ Đã lưu vùng màu xanh hoặc đỏ vào: {cropped_path}")
    return cropped_path

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return "No image part", 400

    file = request.files['image']
    if file.filename == '':
        return "No selected file", 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    print(f"[INFO] Đã nhận ảnh: {file.filename}")

    cropped_path = crop_largest_red_or_blue_region(filepath)
    if not cropped_path:
        return "Không tìm thấy vùng màu xanh hoặc đỏ", 400

    result = random.choice([0, 1, 2])
    print(f"[INFO] Trả về: {result}")
    return str(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
