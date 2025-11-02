import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tkinter import Tk, filedialog, messagebox

# --- Sozlamalar ---
MODEL_PATH = "model/hayvonlar_256_v3.h5"
IMG_SIZE = (256, 256)
CLASS_NAMES = ['kuchuk', 'mushuk', 'ot', 'sichqon', 'sigir']
CONF_THRESHOLD = 70.0  # Agar bashorat ishonchliligi past bo'lsa, boshqalar papkasiga

# --- Tkinter oynasi orqali papka tanlash ---
root = Tk()
root.withdraw()

messagebox.showinfo("📂 Test papkasi", "Iltimos, aralash hayvonlar rasmlari joylashgan papkani tanlang.")
TEST_DIR = filedialog.askdirectory(title="Aralash hayvonlar papkasi")
if not TEST_DIR:
    messagebox.showerror("❌ Xato", "Papka tanlanmadi! Dastur yakunlandi.")
    exit()

messagebox.showinfo("📂 Natija papkasi", "Iltimos, saralangan natijalar joylashadigan papkani tanlang.")
OUTPUT_DIR = filedialog.askdirectory(title="Saralangan hayvonlar papkasi")
if not OUTPUT_DIR:
    messagebox.showerror("❌ Xato", "Natija papkasi tanlanmadi! Dastur yakunlandi.")
    exit()

# --- Modelni yuklash ---
print("🔄 Model yuklanmoqda...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model yuklandi.\n")

# --- Natija papkalarini yaratish va hisoblagichlarni tayyorlash ---
class_counters = {}
for class_name in CLASS_NAMES + ["boshqalar"]:
    class_path = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(class_path, exist_ok=True)
    existing_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    class_counters[class_name] = len(existing_files) + 1

# --- Test papkasi ichidagi barcha rasmlarni rekursiv tarzda topish ---
files = []
for root_dir, _, filenames in os.walk(TEST_DIR):
    for f in filenames:
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            files.append(os.path.join(root_dir, f))

print(f"🔍 {len(files)} ta rasm topildi.\n")

# --- Har bir rasmni aniqlash va ko‘chirish ---
for idx, file_path in enumerate(files, start=1):
    # Rasmni o‘qish
    img = image.load_img(file_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Bashorat qilish
    pred = model.predict(img_array, verbose=0)
    pred_class = np.argmax(pred, axis=1)[0]
    confidence = np.max(pred) * 100

    # Agar ishonchlilik past bo‘lsa, "boshqalar" papkasiga
    if confidence < CONF_THRESHOLD:
        class_name = "boshqalar"
    else:
        class_name = CLASS_NAMES[pred_class]

    # Fayl nomini tartib bilan berish
    dest_folder = os.path.join(OUTPUT_DIR, class_name)
    new_filename = f"img{class_counters[class_name]:03d}.jpg"
    class_counters[class_name] += 1

    # Faylni natija papkasiga ko‘chirish
    shutil.move(file_path, os.path.join(dest_folder, new_filename))

    print(f"[{idx}/{len(files)}] 🧠 {os.path.basename(file_path)} → {class_name.upper()} ({confidence:.1f}%) as {new_filename}")

print("\n✅ Barcha rasmlar saralandi!")
messagebox.showinfo("✅ Tayyor!", f"Barcha rasmlar saralandi!\n📂 Natijalar: {os.path.abspath(OUTPUT_DIR)}")
