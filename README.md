# 🐍 Python Eagle Eye

**Real-time face recognition system from video stream.**  
It watches intently, spots strangers — and never blinks.

**Python Eagle Eye** is a project that demonstrates **real-time face recognition using a webcam**.

The main goal is to distinguish between "known" and "unknown" faces using one of the most accurate and modern face recognition systems as of 2025 — **InsightFace** with the **buffalo_l** model.  
This project can be extended into a home alert system or a "smart watchman".

To work effectively in real time, a CUDA-capable GPU like **GTX 1660 Super** or better is required.  
For recognizing more than 5–10 faces simultaneously, you may need to optimize the code with multithreading, switch to lighter models (like `buffalo_s` or `buffalo_sc`), or use a more powerful **RTX** GPU. You can also process only every N-th frame for better performance.

---

## 🚀 Features

- 💻 Real-time video capture from webcam  
- 🧠 Face recognition using a pre-trained model  
- 📷 Automatic video recording of **unknown** faces  
- 🗃 Manual face database creation via a separate script  
- 🔍 Face detection mode without recognition (for registration)  
- 🔧 Clean project structure ready for extension and customization

---

### ⚙ GPU Driver Note

- **Windows** users: install the official **Game Ready Driver** from NVIDIA.  
- **Linux** users: install both **NVIDIA Driver**, **CUDA Toolkit**, and **cuDNN** for GPU acceleration.

---

## 🏗 Project Structure

```bash
.
├── main.py                 # Start webcam & real-time face recognition
├── record_face.py          # Start webcam to record known face database
├── core/
│   └── recognizer.py       # Core logic for face detection & recognition
├── data/
│   ├── known_faces/        # Folders with images of known people
│   └── unknown_clips/      # Recorded clips of unknown faces
├── poetry.lock             # Dependency lockfile
├── pyproject.toml          # Project dependencies
└── README.md
```