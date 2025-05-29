import os
import cv2
from werkzeug.utils import secure_filename

def process_roboflow_detections(image_path, roboflow_result, output_crop_dir="./crop"):
    """
    根據 Roboflow 的偵測結果裁切圖片，裁切結果會儲存在指定資料夾
    """
    os.makedirs(output_crop_dir, exist_ok=True)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"無法載入圖片：{image_path}")

    height, width = img.shape[:2]
    results = []

    for i, pred in enumerate(roboflow_result.get("predictions", [])):
        x, y = int(pred['x']), int(pred['y'])
        w, h = int(pred['width']), int(pred['height'])

        x1 = max(x - w // 2, 0)
        y1 = max(y - h // 2, 0)
        x2 = min(x + w // 2, width)
        y2 = min(y + h // 2, height)

        if x2 <= x1 or y2 <= y1:
            print(f"⚠️ Skipped invalid box #{i}")
            continue

        crop = img[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
            print(f"⚠️ Skipped too-small crop #{i}: shape={crop.shape}")
            continue

        crop_path = os.path.join(output_crop_dir, f"crop_{i}.jpg")
        cv2.imwrite(crop_path, crop)

        results.append(crop_path)

    return results
