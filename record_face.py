import os
import cv2
import time
import logging
from core.recognizer import FaceRecognizer

# Constants
SAVE_ROOT = "data/known_faces"
SAVE_INTERVAL = 0.2  # seconds

# Logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

def main():
    """
    Launches a webcam-based face capture session for a single person.
    
    The user is prompted to enter a name, and face images are saved in a dedicated
    folder under SAVE_ROOT. Faces are detected using FaceRecognizer, and cropped
    images around the face are saved at regular intervals.
    
    Press ESC to stop recording.
    """
    name = input("Enter a name for face recording: ").strip()
    if not name:
        logger.warning("Name cannot be empty.")
        return

    save_dir = os.path.join(SAVE_ROOT, name)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Failed to open the webcam.")
        return

    recognizer = FaceRecognizer(db_path=None)  # No need to load DB during recording

    logger.info("Starting face capture. Rotate your head. Press ESC to stop.")
    count = 0
    last_save_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to grab frame from webcam.")
            break

        results = recognizer.detect_faces_only(frame)
        current_time = time.time()

        if results and (current_time - last_save_time) >= SAVE_INTERVAL:
            for bbox in results:
                x1, y1, x2, y2 = bbox
                padding = 100
                h, w = frame.shape[:2]
                x1p = max(0, x1 - padding)
                y1p = max(0, y1 - padding)
                x2p = min(w, x2 + padding)
                y2p = min(h, y2 + padding)

                face_img = frame[y1p:y2p, x1p:x2p].copy()
                filename = os.path.join(save_dir, f"{name}_{count+1:03d}.jpg")
                cv2.imwrite(filename, face_img)
                count += 1
                logger.info(f"[{count}] Saved face image: {filename}")

            last_save_time = current_time

        # Draw bounding boxes
        for bbox in results:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        cv2.putText(
            frame,
            "Rotate head slowly in all directions, press ESC to stop",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (100, 255, 100),
            2
        )

        cv2.imshow("Webcam Face Capture", frame)
        key = cv2.waitKey(1)
        if key == 27:  # ESC key
            logger.info("Capture stopped by user.")
            break

    logger.info(f"Capture finished. Total faces saved: {count}")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
