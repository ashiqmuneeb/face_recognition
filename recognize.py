import cv2, joblib, numpy as np, torch
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
from sklearn.preprocessing import normalize
from common_logging import get_logger

logger = get_logger("recognize")

MODELS_DIR = "face_recognition/outputs"
UNKNOWN_THRESHOLD = 0.30   # tune as needed

detector = YOLO(r"D:\face_recognition\face_recognition\models\yolov8n-face.pt")
facenet = InceptionResnetV1(pretrained="vggface2").eval()
knn = joblib.load(f"{MODELS_DIR}/knn_face_classifier.pkl")
le  = joblib.load(f"{MODELS_DIR}/label_encoder.pkl")

def embed_face(face):
    face = cv2.resize(face,(160,160))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    t = torch.tensor(face).permute(2,0,1).unsqueeze(0).float()/255
    with torch.no_grad():
        return normalize(facenet(t).numpy(), axis=1)

def recognize(camera=0):
    logger.info("Starting real-time recognition.")
    cap = cv2.VideoCapture(camera)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logger.error("Camera read failed.")
            break

        results = detector(frame, conf=0.6)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue
                emb = embed_face(face)

                distances, _ = knn.kneighbors(emb, n_neighbors=1)
                nearest = distances[0][0]
                if nearest > UNKNOWN_THRESHOLD:
                    name = "Unknown"
                else:
                    pred = knn.predict(emb)[0]
                    name = le.inverse_transform([pred])[0]

                logger.info(f"Detected: {name}, distance={nearest:.3f}")
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            logger.info("Recognition stopped by user.")
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Recognition finished.")

if __name__ == "__main__":
    recognize()
