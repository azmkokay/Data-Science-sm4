import joblib
import mediapipe as mp
import numpy as np

model = joblib.load("models/face_shape_classifier.pkl")
mp_face_mesh = mp.solutions.face_mesh

# ==================== FUNGSI DISTANCE (SUDAH DI-FIX) ====================
def dist(p1, p2):
    """Jarak antar landmark MediaPipe (pakai .x dan .y)"""
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def get_face_features(landmarks):
    """Ekstrak 8 fitur penting dari landmark wajah"""
    points = {
        'forehead': landmarks[10],
        'chin': landmarks[152],
        'left_cheek': landmarks[234],
        'right_cheek': landmarks[454],
        'left_jaw': landmarks[172],
        'right_jaw': landmarks[397],
        'left_eye_outer': landmarks[33],
        'right_eye_outer': landmarks[263],
        'nose_tip': landmarks[1]
    }
    
    h = dist(points['forehead'], points['chin'])
    w = dist(points['left_cheek'], points['right_cheek'])
    jaw_w = dist(points['left_jaw'], points['right_jaw'])
    eye_dist = dist(points['left_eye_outer'], points['right_eye_outer'])
    
    features = [
        h / w,                    # face ratio
        jaw_w / w,                # jaw vs width
        eye_dist / w,             # eye distance
        dist(points['left_cheek'], points['left_jaw']) / h,
        dist(points['right_cheek'], points['right_jaw']) / h,
        dist(points['forehead'], points['left_cheek']) / w,
        dist(points['forehead'], points['right_cheek']) / w,
        dist(points['chin'], points['nose_tip']) / h
    ]
    return features

def predict_face_shape(landmarks):
    """Prediksi bentuk wajah + confidence"""
    features = get_face_features(landmarks)
    shape = model.predict([features])[0]
    confidence = model.predict_proba([features]).max()
    return shape, confidence