import cv2
import mediapipe as mp

# Inisialisasi MediaPipe
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection()

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert ke RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_detection.process(rgb)

    # Kalau ada wajah
    if result.detections:
        for det in result.detections:
            bbox = det.location_data.relative_bounding_box
            h, w, _ = frame.shape

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)

            # Kotak wajah
            cv2.rectangle(frame, (x,y), (x+width, y+height), (0,255,0), 2)

    cv2.imshow("Deteksi Wajah", frame)

    # Tekan ESC untuk keluar
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()