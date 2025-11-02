# Hayvonlar CNN - Nodirbek Nuraliyev UZBEK AI

Ushbu loyiha **Hayvonlar tasvirlarini tasniflash uchun CNN modeli**ni o‘rganish va takomillashtirish maqsadida ishlab chiqilgan. Model har bir sinf uchun 50 ta rasm (256x256 px) bilan o‘qitilgan va CPUda ham ishlaydi.


Model aniqligi 98%

## 🖼️ Sinflar

- kuchuk
- mushuk
- ot
- sichqon
- sigir

## ⚙️ Ishlatish

1. Loyihani klonlash:

```bash
git clone https://github.com/nuraliyevnodirbek428-source/UZBEK-AI-2025
cd Hayvonlar-CNN


```
# app.py (v7.1 — 5 SINF, 256px, 3 TILLI, XATOSIZ, PROFESSIONAL)
# Nuraliyev Nodirbek 2022-2025 © All rights reserved.
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash, jsonify, session
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import uuid
import logging
from datetime import datetime
import threading
import pyttsx3
import json

# LOGGING
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/app.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
os.makedirs("logs", exist_ok=True)

app = Flask(__name__)
app.secret_key = "uzbek_ai_5sinf_ultra_secure_2025"
app.config.update(
    UPLOAD_FOLDER='static/uploads',
    DATASET_FOLDER='dataset',
    MAX_CONTENT_LENGTH=20 * 1024 * 1024,  # 20MB
    ALLOWED_EXTENSIONS={'png', 'jpg', 'jpeg', 'webp'},
    MODEL_PATH="model/hayvonlar_256_v3.h5",
    IMAGE_SIZE=(256, 256)
)

# TILLAR
LANGUAGES = {
    'uz': {
        'title': "O‘zbekcha Hayvon Aniqlovchi AI",
        'subtitle': "Rasm yuklang → AI darrov aytadi!",
        'analyze': "ANIQLASH",
        'hint': "Faqat .jpg, .jpeg, .png, .webp | 256×256",
        'total': "Jami bashorat",
        'last': "Oxirgi 1 ta",
        'started': "Ishga tushdi",
        'classes': "Sinflar bo‘yicha",
        'result': "Natija:",
        'animal': "Hayvon:",
        'confidence': "Ishonch:",
        'saved': "Saqlangan:",
        'index': "Indeks:",
        'speak': "Ovoz bilan eshittirish",
        'error_file': "Rasm tanlanmadi!",
        'error_type': "Faqat rasm fayllari!",
        'error_process': "Rasmni aniqlab bo‘lmadi!",
        'error_retry': "Qayta urining!",
        'class_names': ['KUCHUK', 'MUSHUK', 'OT', 'SICHQON', 'SIGIR']
    },
    'ru': {
        'title': "Узбекский ИИ Распознаватель Животных",
        'subtitle': "Загрузите фото → ИИ мгновенно определит!",
        'analyze': "РАСПОЗНАТЬ",
        'hint': "Только .jpg, .jpeg, .png, .webp | 256×256",
        'total': "Всего предсказаний",
        'last': "Последние 1",
        'started': "Запущен",
        'classes': "По классам",
        'result': "Результат:",
        'animal': "Животное:",
        'confidence': "Уверенность:",
        'saved': "Сохранено:",
        'index': "Индекс:",
        'speak': "Озвучить результат",
        'error_file': "Фото не выбрано!",
        'error_type': "Только изображения!",
        'error_process': "Не удалось распознать!",
        'error_retry': "Повторите попытку!",
        'class_names': ['СОБАКА', 'КОШКА', 'ЛОШАДЬ', 'МЫШЬ', 'КОРОВА']
    },
    'en': {
        'title': "Uzbek Animal Recognition AI",
        'subtitle': "Upload image → AI instantly detects!",
        'analyze': "ANALYZE",
        'hint': "Only .jpg, .jpeg, .png, .webp | 256×256",
        'total': "Total Predictions",
        'last': "Last 1",
        'started': "Started",
        'classes': "By Class",
        'result': "Result:",
        'animal': "Animal:",
        'confidence': "Confidence:",
        'saved': "Saved:",
        'index': "Index:",
        'speak': "Speak Result",
        'error_file': "No image selected!",
        'error_type': "Image files only!",
        'error_process': "Could not recognize!",
        'error_retry': "Try again!",
        'class_names': ['DOG', 'CAT', 'HORSE', 'MOUSE', 'COW']
    }
}

# OVOZ MOTOR
try:
    tts = pyttsx3.init()
    tts.setProperty('rate', 160)
    tts.setProperty('volume', 1.0)
    voices = tts.getProperty('voices')
    for voice in voices:
        if any(lang in voice.languages for lang in ['uz', 'en', 'ru']):
            tts.setProperty('voice', voice.id)
            break
    OVOZ = True
    logger.info("Ovoz motor tayyor!")
except Exception as e:
    OVOZ = False
    logger.warning(f"Ovoz motor ishlamadi: {e}")

# MODEL YUKLASH
try:
    logger.info(f"Model yuklanmoqda: {app.config['MODEL_PATH']}")
    model = tf.keras.models.load_model(app.config['MODEL_PATH'])
    logger.info("Model yuklandi! 5 sinf: kuchuk, mushuk, ot, sichqon, sigir")
except Exception as e:
    logger.error(f"Model yuklashda xato: {e}")
    model = None

# SINFLAR
CLASSES = ['kuchuk', 'mushuk', 'ot', 'sichqon', 'sigir']
CLASS_UZ = ['KUCHUK', 'MUSHUK', 'OT', 'SICHQON', 'SIGIR']

# PAPKALAR
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
for cls in CLASSES:
    os.makedirs(os.path.join(app.config['DATASET_FOLDER'], cls), exist_ok=True)

# STATISTIKA
STATS_FILE = "stats.json"

def load_stats():
    default = {
        'total': 0,
        'class_count': {cls: 0 for cls in CLASSES},
        'last_predictions': [],
        'start_time': datetime.now().strftime("%Y-%m-%d %H:%M")
    }
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for cls in CLASSES:
                    if cls not in data.get('class_count', {}):
                        data['class_count'][cls] = 0
                data['last_predictions'] = data.get('last_predictions', [])[-10:]
                return data
        except Exception as e:
            logger.error(f"stats.json o'qishda xato: {e}")
    return default

def save_stats():
    try:
        with open(STATS_FILE, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"stats.json saqlashda xato: {e}")

def update_stats(natija):
    stats['total'] += 1
    lower_cls = natija.lower()
    stats['class_count'][lower_cls] = stats['class_count'].get(lower_cls, 0) + 1
    stats['last_predictions'].append(natija)
    if len(stats['last_predictions']) > 1:
        stats['last_predictions'].pop(0)
    save_stats()

stats = load_stats()

# FAYL TEKSHIRISH
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# NOMLASH
def get_next_filename(folder):
    files = [f for f in os.listdir(folder) if f.startswith('img') and f.lower().endswith(('.jpg', '.jpeg'))]
    if not files:
        return "img0001.jpg"
    nums = [int(''.join(filter(str.isdigit, f[3:7]))) for f in files if ''.join(filter(str.isdigit, f[3:7])).isdigit()]
    return f"img{max(nums)+1:04d}.jpg" if nums else "img0001.jpg"

# SAQLASH (256px)
def resize_and_save(src_path, dst_path):
    try:
        with Image.open(src_path) as img:
            img = img.convert("RGB")
            img = img.resize(app.config['IMAGE_SIZE'], Image.Resampling.LANCZOS)
            img.save(dst_path, "JPEG", quality=98, optimize=True)
    except Exception as e:
        logger.error(f"Saqlash xatosi: {e}")

# BASHORAT (256px)
def predict_image(img_path):
    if model is None:
        return "MODEL YO'Q", 0.0
    try:
        img = image.load_img(img_path, target_size=app.config['IMAGE_SIZE'])
        x = image.img_to_array(img) / 255.0
        x = np.expand_dims(x, axis=0)
        pred = model.predict(x, verbose=0)[0]
        idx = np.argmax(pred)
        confidence = pred[idx] * 100
        return CLASS_UZ[idx], round(confidence, 2)
    except Exception as e:
        logger.error(f"Bashorat xatosi: {e}")
        return "XATO", 0.0

# OVOZ (TILGA QARAB)
def speak(text, lang='uz'):
    if OVOZ:
        def run():
            try:
                tts.say(text)
                tts.runAndWait()
            except:
                pass
        threading.Thread(target=run, daemon=True).start()

# STATIK FAYLLAR
@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# API: STATISTIKA
@app.route('/api/stats')
def api_stats():
    return jsonify(stats)

# API: BASHORAT
@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Rasm yo‘q'}), 400
    file = request.files['file']
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'Noto‘g‘ri fayl'}), 400

    ext = file.filename.rsplit('.', 1)[1].lower()
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"api_{uuid.uuid4().hex[:8]}.{ext}")
    file.save(temp_path)

    natija, foiz = predict_image(temp_path)
    os.remove(temp_path)

    if natija not in ["MODEL YO'Q", "XATO"]:
        update_stats(natija)

    return jsonify({
        'natija': natija,
        'foiz': foiz,
        'time': datetime.now().strftime("%H:%M:%S"),
        'class_index': CLASSES.index(natija.lower()) if natija in CLASS_UZ else -1,
        'model': 'hayvonlar_256_v2.h5'
    })

# ASOSIY SAHIFA
@app.route('/', methods=['GET', 'POST'])
def index():
    # TILNI ANIQLASH
    lang = request.args.get('lang', session.get('lang', 'uz'))
    if lang not in LANGUAGES:
        lang = 'uz'
    session['lang'] = lang
    t = LANGUAGES[lang]

    result = {}
    error = None

    if request.method == 'POST':
        if 'file' not in request.files:
            error = 'error_file'
        else:
            file = request.files['file']
            if file.filename == '':
                error = 'error_file'
            elif not allowed_file(file.filename):
                error = 'error_type'
            else:
                try:
                    ext = file.filename.rsplit('.', 1)[1].lower()
                    unique = f"upl_{uuid.uuid4().hex[:10]}.{ext}"
                    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], unique)
                    file.save(upload_path)

                    natija, foiz = predict_image(upload_path)
                    if natija == "XATO":
                        error = 'error_process'
                    else:
                        cls_folder = os.path.join(app.config['DATASET_FOLDER'], natija.lower())
                        filename = get_next_filename(cls_folder)
                        save_path = os.path.join(cls_folder, filename)
                        resize_and_save(upload_path, save_path)

                        display_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                        resize_and_save(upload_path, display_path)

                        update_stats(natija)

                        # OVOZ (TILGA QARAB)
                        class_idx = CLASSES.index(natija.lower())
                        animal_name = t['class_names'][class_idx]
                        speak_texts = {
                            'uz': f"{animal_name}! Ishonch: {foiz} foiz.",
                            'ru': f"{animal_name}! Уверенность: {foiz}%.",
                            'en': f"{animal_name}! Confidence: {foiz}%."
                        }
                        speak(speak_texts[lang], lang)

                        result = {
                            'natija': natija,
                            'foiz': foiz,
                            'img': f'/static/uploads/{filename}',
                            'saved': f"{natija.lower()}/{filename}",
                            'time': datetime.now().strftime("%H:%M:%S"),
                            'animal_name': animal_name
                        }

                        logger.info(f"BASHORAT [{lang}]: {animal_name} ({foiz}%) → {result['saved']}")

                    os.remove(upload_path)

                except Exception as e:
                    logger.error(f"Xato: {e}")
                    error = 'error_retry'

        if error:
            flash(t[error], 'error')

    return render_template('index.html', result=result, stats=stats, t=t, lang=lang)

# XATOLIKLAR
@app.errorhandler(413)
def too_large(e):
    lang = session.get('lang', 'uz')
    t = LANGUAGES[lang]
    flash(t['error_retry'], 'error')
    return redirect(request.url)

# ISHGA TUSHIRISH
if __name__ == '__main__':
    print("="*80)
    print("O‘ZBEKCHA HAYVON ANIQLOVCHI AI v7.1 — 5 SINF 256px — UZ/RU/EN")
    print("SINFLAR: kuchuk(0), mushuk(1), ot(2), sichqon(3), sigir(4)")
    print(f"MODEL: {app.config['MODEL_PATH']}")
    print(f"DATASET: 5 × 40 = 200 rasm")
    print("http://127.0.0.1:5000")
    print("="*80)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)
