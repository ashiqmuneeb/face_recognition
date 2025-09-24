import os, time, cv2, numpy as np, torch
from facenet_pytorch import InceptionResnetV1
from ultralytics import YOLO
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from common_logging import get_logger

logger = get_logger("encode")

MODELS_DIR = 'face_recognition/outputs/'
DATA_PATH  = 'face_recognition/dataset/'

face_detector = YOLO(r'D:\face_recognition\face_recognition\models\yolov8n-face.pt')
facenet = InceptionResnetV1(pretrained="vggface2").eval()

def extract_embedding(face_image):
    try:
        img = cv2.resize(face_image, (160, 160))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.tensor(img).permute(2,0,1).unsqueeze(0).float()/255
        with torch.no_grad():
            return facenet(tensor).numpy().squeeze()
    except Exception as e:
        logger.exception(f"Embedding extraction failed: {e}")
        return None

def create_embeddings(max_images_per_person=30):
    embeddings, labels = [], []
    start_time = time.time()

    for person_name in [d for d in os.listdir(DATA_PATH)
                        if os.path.isdir(os.path.join(DATA_PATH, d))]:
        logger.info(f"Processing {person_name}")
        person_path = os.path.join(DATA_PATH, person_name)
        image_files = [f for f in os.listdir(person_path)
                       if f.lower().endswith(('.png','.jpg','.jpeg'))][:max_images_per_person]
        for img_name in image_files:
            img_path = os.path.join(person_path, img_name)
            image = cv2.imread(img_path)
            results = face_detector(image, conf=0.5)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    face = image[y1:y2, x1:x2]
                    if face.size == 0:
                        continue
                    emb = extract_embedding(face)
                    if emb is not None:
                        embeddings.append(emb)
                        labels.append(person_name)

    if not embeddings:
        logger.error("No embeddings generated.")
        return

    embeddings, labels = np.array(embeddings), np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.2, stratify=labels, random_state=42
    )

    os.makedirs(MODELS_DIR, exist_ok=True)
    np.save(f'{MODELS_DIR}train_embeddings.npy', X_train)
    np.save(f'{MODELS_DIR}test_embeddings.npy',  X_test)
    np.save(f'{MODELS_DIR}train_labels.npy',     y_train)
    np.save(f'{MODELS_DIR}test_labels.npy',      y_test)
    np.save(f'{MODELS_DIR}classes.npy',          np.unique(labels))
    logger.info(f"[SUCCESS] {len(labels)} embeddings created in {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    create_embeddings()
