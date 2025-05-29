import os
import cv2
import numpy as np
import faiss
import json
import tempfile
import shutil
from flask import Flask, request, jsonify, Response, render_template
from werkzeug.utils import secure_filename

from backend.matcher import process_image
from backend.multi_matcher import process_multi_image
from backend.roboflow_api import get_roboflow_predictions
from backend.crop import process_roboflow_detections

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/match', methods=['POST'])
def match():
    category_en = request.form.get("category")
    file = request.files.get("image")

    if not file or not category_en:
        return Response("<p>❌ 缺少類別或圖片</p>", status=400, mimetype='text/html; charset=utf-8')

    img_data = file.read()
    try:
        result_html = process_image(category_en, img_data)
        return Response(result_html, mimetype='text/html; charset=utf-8')
    except Exception as e:
        import traceback
        traceback.print_exc()
        return Response(f"<p>處理錯誤：{str(e)}</p>", status=500, mimetype='text/html; charset=utf-8')

@app.route('/multi_match', methods=['POST'])
def multi_match():
    file = request.files.get("image")
    if not file:
        return Response("<p>❌ 缺少圖片</p>", status=400, mimetype='text/html; charset=utf-8')

    # 從表單或 query string 取得分類
    category_en = request.form.get("category", "all")  # 例如 'effect'、'spell' 等

    # 儲存上傳圖片
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    file.save(image_path)

    # 呼叫 Roboflow 偵測 API
    roboflow_result = get_roboflow_predictions(image_path)

    # 建立暫存資料夾裁切卡片
    temp_dir = tempfile.mkdtemp()
    try:
        cropped_images = process_roboflow_detections(image_path, roboflow_result, output_crop_dir=temp_dir)
        cropped_bytes = []
        for crop_path in cropped_images:
            with open(crop_path, 'rb') as f:
                cropped_bytes.append(f.read())

        # ✅ 傳入 category_en
        result_html = process_multi_image(cropped_bytes, category_en)
        return Response(result_html, mimetype='text/html; charset=utf-8')
    finally:
        shutil.rmtree(temp_dir)
        try:
            os.remove(image_path)
        except Exception as e:
            print(f"⚠️ 無法刪除上傳圖片：{image_path}, 錯誤：{e}")



if __name__ == '__main__':
    app.run(debug=True)
