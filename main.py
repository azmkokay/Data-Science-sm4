import cv2
import mediapipe as mp

# Inisialisasi MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_detection = mp_face_detection.FaceDetection(
    model_selection=0, 
    min_detection_confidence=0.5
)

# Buka kamera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Gagal membuka kamera")
        break

    # Convert ke RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Proses deteksi wajah
    results = face_detection.process(rgb_frame)

    # Jika wajah terdeteksi
    if results.detections:
        for detection in results.detections:
            mp_drawing.draw_detection(frame, detection)

    # Tampilkan hasil
    cv2.imshow("Deteksi Wajah (Python 3.11)", frame)

    # Tekan ESC untuk keluar
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release
cap.release()
cv2.destroyAllWindows()