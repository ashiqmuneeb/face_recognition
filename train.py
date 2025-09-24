from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import numpy as np, joblib, os
from common_logging import get_logger

logger = get_logger("train")
MODELS_DIR = "face_recognition/outputs"

def train_with_validation():
    logger.info("Loading data for training...")
    X_train = normalize(np.load(f"{MODELS_DIR}/train_embeddings.npy"), axis=1)
    X_test  = normalize(np.load(f"{MODELS_DIR}/test_embeddings.npy"),  axis=1)
    y_train = np.load(f"{MODELS_DIR}/train_labels.npy")
    y_test  = np.load(f"{MODELS_DIR}/test_labels.npy")
    classes = np.load(f"{MODELS_DIR}/classes.npy")

    le = LabelEncoder().fit(classes)
    y_train_enc = le.transform(y_train)
    y_test_enc  = le.transform(y_test)

    knn = KNeighborsClassifier(n_neighbors=3, metric="cosine", weights="distance")
    knn.fit(X_train, y_train_enc)
    logger.info("Training finished.")

    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test_enc, y_pred)
    logger.info(f"Validation Accuracy: {acc:.3f}")
    logger.info("\n" + classification_report(y_test_enc, y_pred, target_names=classes))

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(knn, f"{MODELS_DIR}/knn_face_classifier.pkl")
    joblib.dump(le,  f"{MODELS_DIR}/label_encoder.pkl")
    logger.info("[SUCCESS] Model and label encoder saved.")

if __name__ == "__main__":
    train_with_validation()
