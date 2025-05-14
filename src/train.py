import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping


import os

# Lokasi dataset setelah diekstrak
dataset_path = "dataset/fer2013"

# Direktori train dan validasi
train_dir = os.path.join(dataset_path, "train")
val_dir = os.path.join(dataset_path, "validation")

# Parameter
img_size = 48
batch_size = 64

# Data Augmentation pada data training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,           # rotasi gambar max 20 derajat
    width_shift_range=0.1,       # geser horizontal max 10%
    height_shift_range=0.1,      # geser vertikal max 10%
    zoom_range=0.2,              # zoom in/out max 20%
    horizontal_flip=True         # pembalikan horizontal
)

# Validasi tidak perlu augmentasi, hanya rescaling
val_datagen = ImageDataGenerator(rescale=1./255)

# Generator data training
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale'
)

# Generator data validasi
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='grayscale'
)

# Arsitektur CNN
model = Sequential([
    Conv2D(64, (3,3), activation='relu', input_shape=(img_size, img_size, 1)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 kelas emosi
])

# Kompilasi model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Pakai EarlyStopping supaya tidak Overfitting
early_stop = EarlyStopping(
    monitor='val_loss',      # Pantau 'val_loss' agar tidak overfitting
    patience=10,             # Berhenti training jika tidak ada peningkatan selama 10 epoch
    restore_best_weights=True # Kembalikan bobot terbaik setelah training berhenti
)


# Training model
model.fit(
    train_generator,
    epochs=150,
    validation_data=val_generator,
    callbacks=early_stop
)

# Simpan model
model.save("emotion_model.h5")
print("Model berhasil disimpan ke 'emotion_model.h5'")