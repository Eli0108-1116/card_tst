import os
import requests
import zipfile
from flask import Flask, request, Response, render_template
from werkzeug.utils import secure_filename
import tempfile
import shutil

from backend.matcher import process_image
from backend.multi_matcher import process_multi_image
from backend.roboflow_api import get_roboflow_predictions
from backend.crop import process_roboflow_detections

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route("/")
def home():
    return render_template('index.html')

def download_and_extract_cache():
    url = 'https://storage.googleapis.com/card_match_project/cache/cache.zip'  # GCS 公開 URL
    dest_zip = '/app/cache.zip'
    extract_path = '/app/cache/'

    os.makedirs(extract_path, exist_ok=True)
    print("📥 正在下載快取...")
    response = requests.get(url)
    with open(dest_zip, 'wb') as f:
        f.write(response.content)
    print("✅ 下載完成！")
    print("📦 正在解壓縮...")
    with zipfile.ZipFile(dest_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("✅ 解壓縮完成！")

@app.route('/match', methods=['POST'])
def match():
    category_en = request.form.get("category")
    file = request.files.get("image")
    if not file or not category_en:
        return Response("❌ 缺少類別或圖片", status=400)
    img_data = file.read()
    try:
        result_html = process_image(category_en, img_data)
        return Response(result_html, mimetype='text/html')
    except Exception as e:
        return Response(f"❌ 錯誤：{e}", status=500)

@app.route('/multi_match', methods=['POST'])
def multi_match():
    file = request.files.get("image")
    category_en = request.form.get("category", "all")
    if not file:
        return Response("❌ 缺少圖片", status=400)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    file.save(image_path)
    roboflow_result = get_roboflow_predictions(image_path)
    temp_dir = tempfile.mkdtemp()
    try:
        cropped_images = process_roboflow_detections(image_path, roboflow_result, output_crop_dir=temp_dir)
        cropped_bytes = [open(p, 'rb').read() for p in cropped_images]
        result_html = process_multi_image(cropped_bytes, category_en)
        return Response(result_html, mimetype='text/html')
    finally:
        shutil.rmtree(temp_dir)
        os.remove(image_path)

if __name__ == '__main__':
    download_and_extract_cache()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
