import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp

# Load kedua model
geometry_model = joblib.load("models/geometry_model.pkl")
dl_model = load_model("models/efficientnet_model.keras")

mp_face_mesh = mp.solutions.face_mesh

def dist(p1, p2):
    """Jarak antar landmark"""
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def get_face_features(landmarks):
    """Ekstrak 10 fitur (harus sama persis dengan extract_features.py)"""
    points = {
        'forehead': landmarks[10],
        'chin': landmarks[152],
        'left_cheek': landmarks[234],
        'right_cheek': landmarks[454],
        'left_jaw': landmarks[172],
        'right_jaw': landmarks[397],
        'left_eye_outer': landmarks[33],
        'right_eye_outer': landmarks[263],
        'nose_tip': landmarks[1],
        'left_forehead': landmarks[54],      # fitur tambahan
        'right_forehead': landmarks[284]     # fitur tambahan
    }
    
    h = dist(points['forehead'], points['chin'])
    w = dist(points['left_cheek'], points['right_cheek'])
    jaw_w = dist(points['left_jaw'], points['right_jaw'])
    eye_dist = dist(points['left_eye_outer'], points['right_eye_outer'])

    features = [
        h / w,                                      # 1
        jaw_w / w,                                  # 2
        eye_dist / w,                               # 3
        dist(points['left_cheek'], points['left_jaw']) / h,   # 4
        dist(points['right_cheek'], points['right_jaw']) / h, # 5
        dist(points['forehead'], points['left_cheek']) / w,   # 6
        dist(points['forehead'], points['right_cheek']) / w,  # 7
        dist(points['chin'], points['nose_tip']) / h,         # 8
        dist(points['left_forehead'], points['right_forehead']) / w,  # 9
        jaw_w / h                                   # 10
    ]
    return np.array(features, dtype=np.float32).reshape(1, -1)

def predict_face_shape(landmarks):
    """Hybrid Prediction (Geometry + Deep Learning)"""
    features = get_face_features(landmarks)
    
    # Prediksi Geometry ML
    geo_proba = geometry_model.predict_proba(features)[0]
    
    # Prediksi Deep Learning (EfficientNet)
    # Untuk sementara kita pakai rata-rata probabilitas (simple ensemble)
    # nanti bisa di-improve dengan input gambar asli
    dl_proba = dl_model.predict(np.zeros((1, 224, 224, 3)), verbose=0)[0]
    
    # Ensemble: rata-rata probabilitas kedua model
    final_proba = (geo_proba + dl_proba) / 2.0
    final_shape = CLASSES[np.argmax(final_proba)]
    confidence = float(np.max(final_proba))
    
    return final_shape, confidence

# Daftar kelas (wajib ada)
CLASSES = ['heart', 'oblong', 'oval', 'round', 'square']