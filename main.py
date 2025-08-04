import os
import cv2
import time
import logging
from core.recognizer import FaceRecognizer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Constants / configuration
DB_PATH = "data/known_faces"
SAVE_FOLDER = "data/unknown_clips"
RECORD_DURATION = 3  # seconds
THRESHOLD = 1.2
FOURCC = cv2.VideoWriter_fourcc(*'mp4v')


def main():
    """
    Run real-time face recognition using webcam video capture.
    
    - Loads the known faces database from DB_PATH.
    - Detects and identifies faces in real-time with a similarity threshold THRESHOLD.
    - If unknown faces are detected, starts recording video clips for RECORD_DURATION seconds.
    - Draws bounding boxes and labels on detected faces.
    - Press ESC to exit.
    
    Note:
    - If you notice unknown faces are often misclassified as known (false positives),
      consider lowering the THRESHOLD value to increase recognition strictness.
    - Conversely, if known faces are often detected as unknown, try increasing the THRESHOLD.
    - All configurable parameters are defined as uppercase constants at the top of the script.
    """

    recognizer = FaceRecognizer(db_path=DB_PATH)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        logger.error("Failed to open webcam.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    out = None
    recording = False
    record_start_time = 0
    os.makedirs(SAVE_FOLDER, exist_ok=True)

    logger.info("Webcam face recognition started.")

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to read frame from webcam.")
            break

        results = recognizer.identify_faces(frame, threshold=THRESHOLD)
        unknown_found = any(res["name"] == "Unknown" for res in results)

        if unknown_found and not recording:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(SAVE_FOLDER, f"unknown_{timestamp}.mp4")
            out = cv2.VideoWriter(filename, FOURCC, fps, (frame.shape[1], frame.shape[0]))
            recording = True
            record_start_time = time.time()
            logger.info(f"[REC] Recording started: {filename}")

        if recording:
            out.write(frame)
            if time.time() - record_start_time > RECORD_DURATION:
                out.release()
                recording = False
                logger.info("[REC] Recording finished.")

        for res in results:
            x1, y1, x2, y2 = res['bbox']
            name = res['name']
            distance = res['distance']
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            label = f"{name} ({distance:.2f})" if name != "Unknown" else "Unknown"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Webcam Face Recognition", frame)
        if cv2.waitKey(1) == 27:  # ESC key
            logger.info("Exit requested by user.")
            break

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    logger.info("Webcam face recognition terminated.")


if __name__ == "__main__":
    main()
