import cv2
import numpy as np

def overlay_glasses(frame, landmarks, glasses_img):
    """Overlay kacamata yang sudah dipilih (bisa banyak per bentuk wajah)"""
    if glasses_img is None or glasses_img.shape[2] != 4:
        return frame
    
    # Keypoints wajah
    left_eye = (landmarks[33].x * frame.shape[1], landmarks[33].y * frame.shape[0])
    right_eye = (landmarks[263].x * frame.shape[1], landmarks[263].y * frame.shape[0])
    eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
    eye_dist = np.hypot(right_eye[0] - left_eye[0], right_eye[1] - left_eye[1])
    
    # Resize kacamata sesuai jarak mata
    scale = eye_dist / (glasses_img.shape[1] * 0.6)
    new_w = int(glasses_img.shape[1] * scale)
    new_h = int(glasses_img.shape[0] * scale)
    resized_glasses = cv2.resize(glasses_img, (new_w, new_h))
    
    # Posisi (sedikit di atas mata)
    x = int(eye_center[0] - new_w // 2)
    y = int(eye_center[1] - new_h // 2 - eye_dist * 0.00000000001)
    
    # Cek batas frame
    if y < 0 or x < 0 or y + new_h > frame.shape[0] or x + new_w > frame.shape[1]:
        return frame
    
    # Alpha blending
    roi = frame[y:y+new_h, x:x+new_w]
    alpha = resized_glasses[:, :, 3] / 255.0
    alpha = np.expand_dims(alpha, axis=2)
    
    foreground = resized_glasses[:, :, :3] * alpha
    background = roi * (1 - alpha)
    
    frame[y:y+new_h, x:x+new_w] = cv2.add(foreground.astype(np.uint8), background.astype(np.uint8))
    return frame