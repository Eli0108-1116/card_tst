import os
import cv2
import numpy as np
import faiss
import re
import html
from tqdm import tqdm
from backend.image_processing import load_or_build_cache

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INFO_DIR = os.path.join(BASE_DIR, "data", "cards_info_updated")
INDEX_PATH = os.path.join(BASE_DIR, "data", "cache", "all.index")
DESC_FILE = os.path.join(BASE_DIR, "data", "cache", "all.npy")


def build_or_load_index(des_list, desc_dim):
    if os.path.exists(INDEX_PATH):
        return faiss.read_index(INDEX_PATH)
    quantizer = faiss.IndexFlatL2(desc_dim)
    index = faiss.IndexIVFPQ(quantizer, desc_dim, 500, 24, 8)
    index.train(des_list)
    index.add(des_list)
    faiss.write_index(index, INDEX_PATH)
    print("✅ 建立新索引完成")
    return index


def match_single_crop(des1, index, descs, names):
    D, I = index.search(des1.astype('float32'), 2)
    good_per_img = [[] for _ in descs]
    boundaries = np.cumsum([len(d) for d in descs])
    for qi in range(len(des1)):
        d0, d1 = D[qi]
        if d0 < 0.9 * d1:
            tr = int(I[qi, 0])
            idx = np.searchsorted(boundaries, tr, side='right')
            start = boundaries[idx - 1] if idx > 0 else 0
            local = tr - start
            good_per_img[idx].append(cv2.DMatch(qi, local, d0))

    best_idx = max(range(len(good_per_img)), key=lambda i: len(good_per_img[i]), default=None)
    if best_idx is None or len(good_per_img[best_idx]) < 2:
        return None

    return names[best_idx]


def read_info(matched_name, category_en="all"):
    import html, re, os

    card_id = os.path.splitext(matched_name)[0].zfill(8)

    matches = [
        fname for fname in os.listdir(INFO_DIR)
        if fname.startswith(card_id) and fname.lower().endswith(".txt")
    ]
    if not matches:
        print(f"⚠️ 找到相似卡片 {matched_name}，但缺少對應資訊檔案")
        return None

    info_file = os.path.join(INFO_DIR, matches[0])
    print(f"🔍 匹配資訊檔案：{info_file}")

    with open(info_file, encoding="utf-8") as f:
        info = f.read()

    # 🔧 自動組出 GitHub Pages 圖片網址
    BASE_IMAGE_URL = "https://salix5.github.io/query-data/pics"
    img_url = f"{BASE_IMAGE_URL}/{int(card_id)}.jpg"
    img_tag = f'<img src="{img_url}" alt="卡圖" style="max-width:300px; margin:5px;" />'

    # 🔍 移除舊圖片網址、換行格式化
    info = re.sub(r'https?://[^\s"]+\.(?:jpg|jpeg|png)', '', info)
    info = html.escape(info, quote=False).replace("\n", " <br>")

    return f"{img_tag}<br>{info}"


def process_multi_image(image_bytes_list, category_en="all"):
    paths, names, kp_attrs, descs, all_desc = load_or_build_cache("all")
    index = build_or_load_index(all_desc, descs[0].shape[1])

    category_map = {
        'spell': '魔法',
        'trap': '陷阱',
        'effect': '效果',
        'normal': '通常',
        'ritual': '儀式',
        'fusion': '融合',
        'synchro': '同步',
        'xyz': '超量',
        'link': '連結',
        'pendulum': '靈擺',
        'all': '全部'
    }
    category_ch = category_map.get(category_en, category_en)

    results = []
    sift = cv2.SIFT_create()

    for img_data in tqdm(image_bytes_list, desc="處理每張裁切圖片"):
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        kp, des = sift.detectAndCompute(img, None)
        if des is None or len(kp) == 0:
            continue

        matched_name = match_single_crop(des, index, descs, names)
        if matched_name:
            # 讓 read_info 根據 category_en 是不是 all 自動推論分類
            info_html = read_info(matched_name, category_en)
            if info_html:
                results.append(info_html)

    if not results:
        return "<p>❌ 沒有辨識出任何卡片</p>"

    result_text = f"<p>辨識出 {len(results)} 張卡片：</p>"
    for r in results:
        result_text += r + "<hr>"

    return result_text


