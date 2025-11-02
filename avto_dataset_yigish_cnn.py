import os
import requests
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from ddgs import DDGS
from datetime import datetime

# === Sozlamalar ===
MODEL_PATH = "model/hayvonlar_256_v2.h5"
DATASET_FOLDER = "dataset"
TEMP_FOLDER = "temp_downloads"
IMAGE_SIZE = (256, 256)
IMAGES_PER_CLASS = 50
CONFIDENCE_THRESHOLD = 80  # foiz

# Sinflar va qidiruv so‘zlari
SEARCH_TERMS = {
    "kuchuk": ["dog", "собака", "kuchuk"],
    "mushuk": ["cat", "кошка", "mushuk"],
    "ot": ["horse", "лошадь", "ot"],
    "sichqon": ["mouse animal", "мышь животное", "sichqon"],
    "sigir": ["cow", "корова", "sigir"]
}

# === Modelni yuklash ===
print("📦 Model yuklanmoqda...")
model = load_model(MODEL_PATH)
CLASSES = list(SEARCH_TERMS.keys())
print("✅ Model yuklandi! Sinflar:", CLASSES)

# === Papkalarni yaratish ===
os.makedirs(TEMP_FOLDER, exist_ok=True)
for cls in CLASSES + ["boshqalar"]:
    os.makedirs(os.path.join(DATASET_FOLDER, cls), exist_ok=True)

# === Funksiyalar ===
def resize_image(path):
    try:
        img = Image.open(path).convert("RGB")
        img = img.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        img.save(path, "JPEG", quality=95)
        return True
    except:
        if os.path.exists(path):
            os.remove(path)
        return False

def predict_image(model, img_path):
    try:
        img = image.load_img(img_path, target_size=IMAGE_SIZE)
        x = image.img_to_array(img) / 255.0
        x = np.expand_dims(x, axis=0)
        preds = model.predict(x, verbose=0)[0]
        return CLASSES[np.argmax(preds)], np.max(preds)*100
    except:
        return None, 0

def download_images_ddgs(query, max_results=50, folder=TEMP_FOLDER):
    os.makedirs(folder, exist_ok=True)
    downloaded = 0
    with DDGS() as ddgs_search:
        for result in ddgs_search.images(query, max_results=max_results):
            try:
                img_url = result['image']
                ext = os.path.splitext(img_url)[1]
                if ext.lower() not in ['.jpg','.jpeg','.png']:
                    ext = '.jpg'
                img_path = os.path.join(folder, f"{query}_{downloaded}{ext}")
                r = requests.get(img_url, timeout=10)
                with open(img_path, 'wb') as f:
                    f.write(r.content)
                downloaded += 1
            except:
                continue
    print(f"📥 {downloaded} ta rasm yuklandi: {query}")

# === Tanlov menyusi ===
def select_class():
    print("\n🐾 Qaysi sinfdan rasm yig‘ay?")
    for i, cls in enumerate(CLASSES, start=1):
        print(f"{i}. {cls}")
    print(f"{len(CLASSES)+1}. BARCHASI")
    while True:
        try:
            choice = int(input("Tanlang (raqam): "))
            if 1 <= choice <= len(CLASSES):
                return [CLASSES[choice-1]]
            elif choice == len(CLASSES)+1:
                return CLASSES
        except:
            pass
        print("❌ Noto‘g‘ri tanlov, qayta kiriting!")

# === Asosiy jarayon ===
while True:
    selected = select_class()
    for cls in selected:
        print(f"\n==== 🐾 SINF: {cls.upper()} ====")
        for term in SEARCH_TERMS[cls]:
            print(f"🔎 Qidirilmoqda: {term}")
            download_images_ddgs(term, IMAGES_PER_CLASS // len(SEARCH_TERMS[cls]))

        files = os.listdir(TEMP_FOLDER)
        valid_count = 0

        for f in files:
            fpath = os.path.join(TEMP_FOLDER, f)
            if not os.path.exists(fpath):
                continue  # mavjud bo'lmagan faylni o'tkazib yubor

            if resize_image(fpath):
                pred_class, confidence = predict_image(model, fpath)
                if pred_class == cls and confidence >= CONFIDENCE_THRESHOLD:
                    dst = os.path.join(DATASET_FOLDER, cls, f)
                else:
                    dst = os.path.join(DATASET_FOLDER, "boshqalar", f)
                os.replace(fpath, dst)
                valid_count += 1
                print(f"🖼️ {f} → {os.path.basename(os.path.dirname(dst)).upper()} ({confidence:.1f}%)")

        print(f"✅ {cls} sinfi uchun {valid_count} ta rasm saqlandi.")

    ans = input("\n👉 Yana davom etaymi? (ha/yo‘q): ").strip().lower()
    if ans != "ha":
        print("🚀 Tugatildi! Dataset yangilandi:", datetime.now().strftime("%Y-%m-%d %H:%M"))
        break
