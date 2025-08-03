import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis


class FaceRecognizer:
    def __init__(self, db_path):
        self.db_path = db_path
        self.app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.database = self.load_database()

    def load_database(self):
        db = {}
        for person in os.listdir(self.db_path):
            person_path = os.path.join(self.db_path, person)
            if not os.path.isdir(person_path):
                continue
            embeddings = []
            for img_file in os.listdir(person_path):
                img_path = os.path.join(person_path, img_file)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                faces = self.app.get(img)
                if faces:
                    embeddings.append(faces[0].embedding)
            if embeddings:
                db[person] = embeddings
        return db

    def identify_faces(self, frame):
        faces = self.app.get(frame)
        results = []

        for face in faces:
            emb = face.embedding
            box = face.bbox.astype(int)
            x1, y1, x2, y2 = box
            identity = "Unknown"
            min_dist = 1.0

            for name, db_embeds in self.database.items():
                for db_emb in db_embeds:
                    dist = np.linalg.norm(emb - db_emb)
                    if dist < min_dist:
                        min_dist = dist
                        identity = name

            if min_dist < 0.6:
                results.append({"name": identity, "distance": min_dist, "bbox": (x1, y1, x2, y2)})
            else:
                results.append({"name": "Unknown", "distance": min_dist, "bbox": (x1, y1, x2, y2)})

        return results
