import os
import cv2
from datetime import datetime

UNKNOWN_DIR = "data/unknown_faces"
os.makedirs(UNKNOWN_DIR, exist_ok=True)


def save_unknown_face(frame, bbox):
    x1, y1, x2, y2 = bbox
    face_img = frame[y1:y2, x1:x2]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{timestamp}.jpg"
    path = os.path.join(UNKNOWN_DIR, filename)
    cv2.imwrite(path, face_img)
