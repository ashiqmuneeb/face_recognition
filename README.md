# Face Recognition with YOLOv8 + FaceNet + KNN

A real-time, multi-person face-recognition system in Python.  
It detects faces with **YOLOv8**, generates **FaceNet** embeddings,  
and classifies known individuals using a **K-Nearest-Neighbors (KNN)** model.  
Unseen faces are automatically labelled **“Unknown”** using a distance threshold.

---

## ✨ Features
- **Real-time detection** of multiple faces from webcam or IP camera.
- **High-accuracy embeddings** with a pre-trained FaceNet model.
- **Distance-threshold rejection** so strangers are marked *Unknown*.
- **Dataset tools** to capture images and create embeddings.
- **Unified logging** with rotating log file for easier debugging.

---

> **Note**: `models/`, `dataset/`, `outputs/`, and `logs/` folders are excluded  
> (see `.gitignore`) because they contain large or generated files.

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/ashiqmuneeb/face_recognition.git
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
