import os
import numpy as np
import cv2
import faiss
import re
from backend.image_processing import load_or_build_cache

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INFO_DIR = os.path.join(BASE_DIR, "data", "cards_info")

def process_image(category_en, img_data):
    # 英文 → 中文（只用於組圖檔網址）
    category_map = {
        'spell': '魔法',
        'trap': '陷阱',
        'effect': '效果',
        'normal': '通常',
        'ritual': '儀式',
        'fusion': '融合',
        'synchro': '同步',
        'xyz': '超量',0000
        'link': '連結',
        'pendulum': '靈擺',
        'all': '全部'
    }
    category_ch = category_map.get(category_en, category_en)

    sift = cv2.SIFT_create()
    img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError("❌ 無法讀取上傳的圖像")

    kp1, des1 = sift.detectAndCompute(img, None)
    if des1 is None or len(kp1) == 0:
        raise ValueError("❌ 圖像中找不到足夠的特徵點")

    d = des1.shape[1]

    # ❗ 這裡使用英文類別名稱來載入 cache
    paths, names, kp_attrs, descs, all_desc = load_or_build_cache(category_en)

    index_path = os.path.join(os.path.dirname(INFO_DIR), "cache", f"{category_en}.index")
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
    else:
        quantizer = faiss.IndexFlatL2(d)
        nlist, m_pq = 100, 16
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m_pq, 8)
        index.train(all_desc)
        index.add(all_desc)
        faiss.write_index(index, index_path)

    index.nprobe = 1

    D, I = index.search(des1.astype('float32'), 2)
    good_per_img = [[] for _ in descs]
    boundaries = np.cumsum([len(d) for d in descs])
    for qi in range(len(des1)):
        d0, d1 = D[qi]
        if d0 < 0.85 * d1:
            tr = int(I[qi, 0])
            idx = np.searchsorted(boundaries, tr, side='right')
            start = boundaries[idx - 1] if idx > 0 else 0
            local = tr - start
            good_per_img[idx].append(cv2.DMatch(qi, local, d0))

    best_idx = max(range(len(good_per_img)), key=lambda i: len(good_per_img[i]), default=None)
    if best_idx is None or len(good_per_img[best_idx]) == 0:
        return "<p>❌ 沒有找到匹配的卡片</p>"

    matched_name = names[best_idx]
    card_id = matched_name[:8]

    matches = [
        fname for fname in os.listdir(INFO_DIR)
        if fname.startswith(card_id) and fname.lower().endswith(".txt")
    ]

    if not matches:
        return f"<p>⚠️ 找到相似卡片 {matched_name}，但缺少對應資訊檔</p>"

    info_file = os.path.join(INFO_DIR, matches[0])
    print(f"🔍 匹配資訊檔案：{info_file}")

    with open(info_file, encoding="utf-8") as f:
        info = f.read()

    # ✅ 移除舊圖片（img 標籤）和 舊圖片 URL 行
    info = re.sub(r'<img.*?>', '', info)  # 移除 HTML 圖片標籤
    info = re.sub(r'圖片 URL:.*?(\r?\n|<br>|$)', '', info)  # 移除圖片 URL 行（支援換行或 <br>）

    # ✅ 中文類別只用來組圖片路徑
    image_url = f"https://res.cloudinary.com/dbqy3zmvq/image/upload/ygocards/{category_ch}/{category_ch}/{card_id}.jpg"

    info_html = f'''
        <p>卡片圖片</p>
        <img src="{image_url}" alt="卡片圖片" style="max-width: 300px;" />
        <br>
        <a href="{image_url}" target="_blank">🔗 開啟圖片連結</a>
        <br><br>
    '''

    # ✅ 換行處理
    info = info.replace("\n", "<br>")

    return info_html + info






