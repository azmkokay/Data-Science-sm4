import cv2
import mediapipe as mp
import os
from face_shape_detector import predict_face_shape
from glasses_overlay import overlay_glasses

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5, refine_landmarks=True)

# ====================== LOAD SEMUA FRAME KACAMATA ======================
GLASSES_DIR = "assets/glasses_frames"
available_frames = {}      # shape -> {nama_frame: image}
frame_lists = {}           # shape -> [nama1, nama2, ...]
current_indices = {}       # shape -> index saat ini

for shape_folder in os.listdir(GLASSES_DIR):
    shape_path = os.path.join(GLASSES_DIR, shape_folder)
    if os.path.isdir(shape_path):
        shape = shape_folder.lower()
        available_frames[shape] = {}
        for filename in os.listdir(shape_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                frame_name = os.path.splitext(filename)[0]
                img = cv2.imread(os.path.join(shape_path, filename), cv2.IMREAD_UNCHANGED)
                if img is not None and img.shape[2] == 4:
                    available_frames[shape][frame_name] = img
        
        if available_frames[shape]:
            frame_lists[shape] = list(available_frames[shape].keys())
            current_indices[shape] = 0

print(f"✅ Berhasil load {sum(len(v) for v in available_frames.values())} frame kacamata!")

# ====================== WEBCAM ======================
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

print("🎥 Webcam aktif. Tekan 'Q' keluar | Tekan 'N' ganti frame kacamata")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    
    current_shape = None
    current_frame_name = ""
    
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        shape, conf = predict_face_shape(landmarks)
        current_shape = shape
        
        if shape in frame_lists and frame_lists[shape]:
            idx = current_indices[shape]
            current_frame_name = frame_lists[shape][idx]
            glasses_img = available_frames[shape][current_frame_name]
            
            # Overlay
            frame = overlay_glasses(frame, landmarks, glasses_img)
            
            # Tampilkan teks
            cv2.putText(frame, f"Bentuk: {shape.upper()} ({conf:.2f})", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(frame, f"Frame: {current_frame_name}", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    cv2.putText(frame, "Tekan N → ganti frame | Q → keluar", (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    
    cv2.imshow("Virtual Glasses Try-On - Multi Frame", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q'):
        break
    if key == ord('n') or key == ord('N'):
        if current_shape and current_shape in current_indices:
            current_indices[current_shape] = (current_indices[current_shape] + 1) % len(frame_lists[current_shape])

cap.release()
cv2.destroyAllWindows()