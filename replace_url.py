import os

# ====== Cloudinary 帳號名稱 ======
cloud_name = 'dbqy3zmvq'

# ====== 原始與輸出資料夾 ======
input_folder = 'data/cards_info'
output_folder = 'data/cards_info_updated'

# 確保輸出資料夾存在
os.makedirs(output_folder, exist_ok=True)

def replace_url_in_file(file_path, output_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    card_id = None
    category_ch = None
    new_lines = []

    for line in lines:
        if line.startswith('卡號:'):
            card_id = line.split(':', 1)[1].strip()
        elif line.startswith('類型:'):
            type_full = line.split(':', 1)[1].strip()
            category_ch = type_full.split('/')[0].strip()

    if not card_id:
        card_id = 'unknown'
    if not category_ch:
        category_ch = 'unknown'

    for line in lines:
        if line.startswith('圖片 URL:'):
            new_url = f'https://res.cloudinary.com/{cloud_name}/image/upload/ygocards/{category_ch}/{category_ch}/{card_id}.jpg'
            new_lines.append(f'圖片 URL: {new_url}\n')
        else:
            new_lines.append(line)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

# 處理整個資料夾
count = 0
for filename in os.listdir(input_folder):
    if filename.endswith('.txt'):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        replace_url_in_file(input_path, output_path)
        count += 1

print(f"✅ 已處理 {count} 筆卡片檔案，結果儲存在：{output_folder}")


    

