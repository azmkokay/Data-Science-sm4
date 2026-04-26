import cv2
import numpy as np

def overlay_glasses(frame, landmarks, glasses_img):
    """Overlay kacamata dengan ROTASI + Alpha yang BENAR (anti terpotong)"""
    if glasses_img is None or glasses_img.shape[2] != 4:
        return frame

    # 1. Ambil posisi mata
    left_eye  = np.array([landmarks[33].x * frame.shape[1], landmarks[33].y * frame.shape[0]])
    right_eye = np.array([landmarks[263].x * frame.shape[1], landmarks[263].y * frame.shape[0]])

    # 2. Hitung sudut rotasi
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = -np.degrees(np.arctan2(dy, dx))

    # 3. Ukuran & skala
    eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
    eye_dist = np.hypot(dx, dy)

    scale = eye_dist / (glasses_img.shape[1] * 0.60)  # untuk mengatur lebar frame
    new_w = int(glasses_img.shape[1] * scale)
    new_h = int(glasses_img.shape[0] * scale)

    # Resize dulu
    resized = cv2.resize(glasses_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 4. Tambah padding BESAR supaya tidak terpotong saat rotasi
    pad = int(max(new_w, new_h) * 0.6)          # ← ini yang paling penting
    padded = cv2.copyMakeBorder(resized, pad, pad, pad, pad,
                               cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))

    # 5. Rotasi
    center = (padded.shape[1] // 2, padded.shape[0] // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    rotated = cv2.warpAffine(padded, rot_matrix, 
                             (padded.shape[1], padded.shape[0]),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(0, 0, 0, 0))

    # 6. Crop padding kembali
    rotated = rotated[pad:-pad, pad:-pad]

    # 7. Posisi akhir
    x = int(eye_center[0] - rotated.shape[1] // 2)
    y = int(eye_center[1] - rotated.shape[0] // 2 - eye_dist * 0.01) # untuk mengatur jarak vertikal

    # Cek batas
    if y < 0 or x < 0 or y + rotated.shape[0] > frame.shape[0] or x + rotated.shape[1] > frame.shape[1]:
        return frame

    # 8. Alpha blending
    roi = frame[y:y+rotated.shape[0], x:x+rotated.shape[1]]
    alpha = rotated[:, :, 3] / 255.0
    alpha = np.expand_dims(alpha, axis=2)

    foreground = rotated[:, :, :3] * alpha
    background = roi * (1 - alpha)

    frame[y:y+rotated.shape[0], x:x+rotated.shape[1]] = cv2.add(
        foreground.astype(np.uint8), 
        background.astype(np.uint8)
    )

    return frame