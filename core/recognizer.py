"""
FaceRecognizer module ("Zorky Python").

This module provides face detection and recognition functionality using InsightFace
and the "buffalo_l" model with GPU acceleration via CUDA.

Features:
- Load face embeddings from a structured directory (each subfolder = one person).
- Identify faces on camera frames based on cosine distance of embeddings.
- Return bounding boxes and identities for recognized/unknown faces.

Example directory structure for known faces:
    data/known_faces/
        john/
            1.jpg
            2.jpg
        anna/
            1.jpg
            2.jpg

Dependencies:
- insightface
- opencv-python
- numpy
"""

import os
import cv2
import numpy as np
import logging
from insightface.app import FaceAnalysis

# Logger setup
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)


class FaceRecognizer:
    """
    A class for face detection and recognition using InsightFace (buffalo_l model).

    Attributes:
        db_path (str): Path to the folder containing known faces.
        app (FaceAnalysis): Initialized face analysis engine.
        database (dict): Dictionary of person names and their face embeddings.
    """

    def __init__(self, db_path: str):
        """
        Initialize the FaceRecognizer.

        Args:
            db_path (str): Path to the folder with known faces.
        """
        self.db_path = db_path
        self.app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        self.database = {}
        if self.db_path:
            self.database = self.load_database()
            logger.info(f"[Init] Database loaded: {len(self.database)} person(s)")

    def load_database(self) -> dict:
        """
        Load known face embeddings from the database folder.

        Returns:
            dict: Dictionary where keys are person names and values are lists of embeddings.
        """
        db = {}
        logger.info(f"Loading database from: {self.db_path}")
        persons = os.listdir(self.db_path)
        logger.info(f"Found directories: {persons}")

        for person in persons:
            person_path = os.path.join(self.db_path, person)
            if not os.path.isdir(person_path):
                continue

            logger.info(f"Loading embeddings for: {person}")
            embeddings = []
            imgs = os.listdir(person_path)
            logger.info(f"  Image files found: {len(imgs)}")

            for img_file in imgs:
                img_path = os.path.join(person_path, img_file)
                img = cv2.imread(img_path)
                if img is None:
                    logger.warning(f"  [WARN] Failed to load image: {img_file}")
                    continue

                faces = self.app.get(img)
                if not faces:
                    logger.warning(f"[WARN] No face detected in: {img_file}")
                    continue

                embedding = faces[0].embedding
                embedding = embedding / np.linalg.norm(embedding)
                embeddings.append(embedding)

            if embeddings:
                db[person] = embeddings
                logger.info(f"[OK] {person}: {len(embeddings)} embeddings added")

        return db

    def identify_faces(self, frame: np.ndarray, threshold: float = 1.206) -> list:
        """
        Identify known or unknown faces in the given frame.

        Args:
            frame (np.ndarray): BGR image from OpenCV.
            threshold (float): Distance threshold for face recognition.

        Returns:
            list: List of dicts containing name, distance, and bounding box for each face.
        """
        faces = self.app.get(frame)
        results = []

        for face in faces:
            emb = face.embedding
            emb = emb / np.linalg.norm(emb)

            box = face.bbox.astype(int)
            x1, y1, x2, y2 = box

            identity = "Unknown"
            min_dist = float("inf")
            best_match = None

            for name, db_embeds in self.database.items():
                distances = [np.linalg.norm(emb - db_emb) for db_emb in db_embeds]
                min_d = min(distances)
                if min_d < min_dist:
                    min_dist = min_d
                    best_match = name

            if min_dist < threshold:
                results.append({
                    "name": best_match,
                    "distance": min_dist,
                    "bbox": (x1, y1, x2, y2)
                })
            else:
                results.append({
                    "name": "Unknown",
                    "distance": min_dist,
                    "bbox": (x1, y1, x2, y2)
                })

        return results

    def detect_faces_only(self, frame: np.ndarray) -> list:
        """
        Detect faces in the frame and return their bounding boxes.

        Args:
            frame (np.ndarray): BGR image from OpenCV.

        Returns:
            list: List of bounding boxes (x1, y1, x2, y2) for detected faces.
        """
        faces = self.app.get(frame)
        return [face.bbox.astype(int) for face in faces]
