# face_shape_simple.py - Versi sederhana tanpa MediaPipe
import cv2
import numpy as np

class SimpleFaceShapeAnalyzer:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def detect_face_shape(self, image_path):
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
        
        if len(faces) == 0:
            return "Tidak ada wajah terdeteksi"
        
        for (x, y, w, h) in faces:
            # Hitung rasio aspek wajah
            aspect_ratio = h / w
            
            if aspect_ratio > 1.3:
                shape = "Panjang (Long)"
            elif aspect_ratio < 1.1:
                shape = "Bulat"
            else:
                shape = "Oval"
            
            # Gambar bounding box
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, shape, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Face Shape', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            return shape

if __name__ == "__main__":
    analyzer = SimpleFaceShapeAnalyzer()
    result = analyzer.detect_face_shape("path_to_your_image.jpg")
    print(f"Bentuk wajah: {result}")