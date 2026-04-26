from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.ensemble import RandomForestClassifier
import joblib

TRAIN_IMG_DIR = "dataset/training_set"
CLASSES = ['heart', 'oblong', 'oval', 'round', 'square']
IMG_SIZE = (224, 224)

os.makedirs("models", exist_ok=True)

# ==================== 1. Train Geometry Model ====================
def train_geometry():
    print("🚀 Training Geometry Model (RandomForest)...")
    df = pd.read_csv("dataset/train_features.csv")
    X = df.iloc[:, :-1].values
    y = df['label'].values

    model = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)
    model.fit(X, y)
    joblib.dump(model, "models/geometry_model.pkl")
    print("✅ Geometry model selesai")

# ==================== 2. Train Deep Learning + Plot Grafik ====================
def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.legend()

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training & Validation Loss')
    plt.legend()

    plt.suptitle('EfficientNetB0 Training History')
    plt.tight_layout()
    
    # Simpan grafik
    plt.savefig("models/training_history.png", dpi=300)
    plt.show()
    print("📊 Grafik training telah disimpan di: models/training_history.png")

def train_dl():
    print("🚀 Training EfficientNetB0 (Deep Learning)...")
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_gen = datagen.flow_from_directory(
        TRAIN_IMG_DIR, target_size=IMG_SIZE, batch_size=32,
        class_mode='categorical', subset='training'
    )
    val_gen = datagen.flow_from_directory(
        TRAIN_IMG_DIR, target_size=IMG_SIZE, batch_size=32,
        class_mode='categorical', subset='validation'
    )

    base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(len(CLASSES), activation='softmax')(x)

    model = Model(base.input, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("Training dimulai...")
    history = model.fit(
        train_gen, 
        validation_data=val_gen, 
        epochs=40, 
        verbose=1
    )
    
    model.save("models/efficientnet_model.keras")
    print("✅ EfficientNet selesai training")
    
    # Tampilkan grafik
    plot_training_history(history)

if __name__ == "__main__":
    train_geometry()
    train_dl()
    print("🎉 Hybrid training selesai! Semua model dan grafik sudah disimpan.")