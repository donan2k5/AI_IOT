import numpy as np
import cv2
from PIL import Image
from keras.models import load_model
import os

# Load model
model = load_model('my_model.h5')

# Nhãn đầu ra
target_classes = {
    14: 0,  # stop
    33: 2,  # right
    34: 1   # left
}

def zoom_crop_expand(image_path, zoom=3.0, padding=35):  # Đổi padding ở đây
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Không tìm thấy ảnh: {image_path}")
        return None

    h, w = img.shape[:2]
    new_w, new_h = int(w * zoom), int(h * zoom)
    img_zoomed = cv2.resize(img, (new_w, new_h))
    x_start = (new_w - w) // 2
    y_start = (new_h - h) // 2
    img_cropped = img_zoomed[y_start:y_start + h, x_start:x_start + w]

    hsv = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2HSV)

    # Màu đỏ
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    # Màu xanh dương
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # Mask
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.bitwise_or(mask_red, mask_blue)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w_box, h_box = cv2.boundingRect(max(contours, key=cv2.contourArea))

        # Lùi lại một khoảng padding
        x = max(0, x - padding)
        y = max(0, y - padding)
        x2 = min(img_cropped.shape[1], x + w_box + 2 * padding)
        y2 = min(img_cropped.shape[0], y + h_box + 2 * padding)

        cropped = img_cropped[y:y2, x:x2]
    else:
        print("⚠ Không tìm thấy vùng đỏ hoặc xanh dương.")
        cropped = img_cropped

    # Convert sang RGB để dùng với PIL
    cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cropped_rgb)

def predict_traffic_sign(image_path):
    try:
        if not os.path.exists(image_path):
            print(f"File not found: {image_path}")
            return -1

        print(f"Processing image: {image_path}")
        processed_img = zoom_crop_expand(image_path, zoom=3.0, padding=35)  # Đổi padding ở đây luôn
        if processed_img is None:
            return -1

        image = processed_img.convert('RGB').resize((30, 30))
        image = np.expand_dims(np.array(image), axis=0)

        pred_index = np.argmax(model.predict(image)[0])
        if pred_index in target_classes:
            result = target_classes[pred_index]
            print(f"Model predicted index: {pred_index} -> Output: {result}")
            return result
        else:
            print(f"Model predicted index: {pred_index} -> Not a recognized sign")
            return -1

    except Exception as e:
        print(f"Error processing image: {e}")
        return -1

# Chạy thử
if __name__ == "__main__":
    image_path = "D:\Study\Python\Iotfpt\image\ANH\Anh_PI\image.jpg"
    result = predict_traffic_sign(image_path)
    print(f"Final Prediction: {result}")
