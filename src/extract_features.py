import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

def dist(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def get_face_features(landmarks):
    # Landmark indices MediaPipe Face Mesh
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

# Proses dataset
def process_dataset(folder):
    X, y = [], []
    shapes = ['heart', 'oblong', 'oval', 'round', 'square']
    for shape in shapes:
        path = os.path.join(folder, shape)
        for img_file in tqdm(os.listdir(path), desc=shape):
            img_path = os.path.join(path, img_file)
            image = cv2.imread(img_path)
            if image is None: continue
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            if results.multi_face_landmarks:
                lm = results.multi_face_landmarks[0].landmark
                landmarks = [(lm[i].x, lm[i].y) for i in range(468)]
                features = get_face_features(landmarks)
                X.append(features)
                y.append(shape)
    return np.array(X), np.array(y)

print("Processing TRAIN...")
X_train, y_train = process_dataset("data/train")
print("Processing TEST...")
X_test, y_test = process_dataset("data/test")

df_train = pd.DataFrame(X_train)
df_train['label'] = y_train
df_train.to_csv("data/train_features.csv", index=False)

df_test = pd.DataFrame(X_test)
df_test['label'] = y_test
df_test.to_csv("data/test_features.csv", index=False)

print("✅ Feature extraction selesai!")