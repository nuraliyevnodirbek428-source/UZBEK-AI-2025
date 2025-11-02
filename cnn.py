import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import json
import os

# --- CPU majburlash ---
tf.config.set_visible_devices([], 'GPU')
print("⚠️ CPU ishlatiladi!")

# --- Dataset ---
dataset_path = "hayvonlar"  # har bir sinfda 50 rasm
num_classes = 5

# --- Data augmentation ---
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=40,
    width_shift_range=0.25,
    height_shift_range=0.25,
    zoom_range=0.25,
    horizontal_flip=True,
    brightness_range=(0.7,1.3),
    shear_range=0.2,
    fill_mode='nearest'
)

img_size = (256, 256)
batch_size = 8

train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# --- MobileNetV2 Base ---
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(256, 256, 3)
)

# Fine-tuning: so'nggi 20 qatlamni trainable qilamiz
for layer in base_model.layers[:-20]:
    layer.trainable = False
for layer in base_model.layers[-20:]:
    layer.trainable = True

# --- Model ---
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(num_classes, activation='softmax')
])

# --- Compile ---
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --- Callbacks ---
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=12,
    restore_best_weights=True
)
reduce_lr = ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

# --- Train ---
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=50,
    verbose=1,
    callbacks=[early_stop, reduce_lr]
)

# --- Evaluate ---
loss, acc = model.evaluate(val_data)
print(f"\n✅ Validation accuracy: {acc*100:.2f}%")

# --- Save model ---
os.makedirs("model", exist_ok=True)
model.save("model/hayvonlar_256_v3_max_cpu.h5")

# --- Save class names ---
class_indices = train_data.class_indices
with open("model/class_names_v3_max_cpu.json", "w", encoding="utf-8") as f:
    json.dump(class_indices, f, ensure_ascii=False, indent=2)

print("\n📁 Model va sinf nomlari saqlandi.")
