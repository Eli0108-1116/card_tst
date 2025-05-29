import os
import numpy as np
import cv2
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = "/app/cache"  # GCS 下載快取
GALLERY_DIR = os.path.join(BASE_DIR, "data", "gallery")  # 本地圖片庫


def extract_features(image_path):
    sift = cv2.SIFT_create()
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ 無法讀取圖片：{image_path}")
        return None, None
    kp, des = sift.detectAndCompute(img, None)
    return kp, des

def build_cache(category):
    gallery_path = os.path.join(GALLERY_DIR, category)
    cache_file = os.path.join(CACHE_DIR, f"{category}.npz")
    desc_file = os.path.join(CACHE_DIR, f"{category}.npy")

    if not os.path.exists(gallery_path):
        raise FileNotFoundError(f"圖片目錄不存在：{gallery_path}")

    print(f"🔨 正在建立快取：{category}")
    paths, names, kp_attrs, descs = [], [], [], []
    sift = cv2.SIFT_create()

    file_list = os.listdir(gallery_path)
    for fname in tqdm(file_list, desc=f"提取特徵中 ({category})", unit="張"):
        img_path = os.path.join(gallery_path, fname)
        kp, des = extract_features(img_path)
        if des is not None:
            paths.append(img_path)
            names.append(fname)
            kp_attrs.append([(p.pt[0], p.pt[1], p.size, p.angle) for p in kp])
            descs.append(des)

    print("💾 儲存壓縮快取 (.npz)...")
    savez = {
        'paths': np.array(paths),
        'names': np.array(names),
        'kp_attrs': np.array(kp_attrs, dtype=object)
    }
    for i, des in enumerate(descs):
        savez[f'des{i}'] = des
    np.savez_compressed(cache_file, **savez)

    if descs:
        print("💾 儲存所有描述子 (.npy)...")
        np.save(desc_file, np.vstack(descs).astype('float32'))
        print(f"✅ 快取構建完成：{category}")
    else:
        print(f"⚠️ 無法建立快取：{category} 沒有有效的圖像數據")

def load_or_build_cache(category):
    cache_file = os.path.join(CACHE_DIR, f"{category}.npz")
    desc_file = os.path.join(CACHE_DIR, f"{category}.npy")

    if os.path.exists(cache_file) and os.path.exists(desc_file):
        print(f"✅ 已加載快取：{category}")
        npz = np.load(cache_file, allow_pickle=True)
        paths = npz['paths'].tolist()
        names = npz['names'].tolist()
        kp_attrs = npz['kp_attrs']
        descs = [npz[f'des{i}'] for i in range(len(names))]
        all_desc = np.load(desc_file)
        return paths, names, kp_attrs, descs, all_desc

    print(f"⚠️ 快取不存在，開始構建：{category}")
    build_cache(category)

    if os.path.exists(cache_file) and os.path.exists(desc_file):
        return load_or_build_cache(category)

    print(f"❌ 無法建立快取：{category}")
    return None, None, None, None, None

