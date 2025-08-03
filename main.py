import os
import cv2
from core.recognizer import FaceRecognizer
from core.face_manager import save_unknown_face

# Путь к видеофайлу
VIDEO_PATH = "miss-russia.mp4"

# Создание экземпляра распознавания лиц
recognizer = FaceRecognizer(db_path="data/known_faces")

# Чтение видео
cap = cv2.VideoCapture(VIDEO_PATH)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Получение результатов распознавания
    results = recognizer.identify_faces(frame)

    for res in results:
        x1, y1, x2, y2 = res['bbox']
        name = res['name']
        confidence = res['distance']

        if name != "Unknown":
            color = (0, 255, 0)
            label = f"{name} ({confidence:.2f})"
        else:
            color = (0, 0, 255)
            label = "Unknown"
            save_unknown_face(frame, (x1, y1, x2, y2))

        # Отрисовка прямоугольника и подписи
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Отображение окна
    cv2.imshow("Recognition", frame)
    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()