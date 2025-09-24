from ultralytics import YOLO
import cv2, os
from PIL import Image
from common_logging import get_logger

logger = get_logger("capture")

# Load YOLO face detector – adjust path if needed
model = YOLO(r'D:\face_recognition\face_recognition\models\yolov8n-face.pt')

def capture_faces(ip_camera=0, num_images=50):
    """Capture images of ALL faces in frame."""
    person_name = input("Enter person's name: ").strip()
    if not person_name:
        logger.warning("No name given. Aborting.")
        return

    person_dir = os.path.join("face_recognition", "dataset", person_name)
    os.makedirs(person_dir, exist_ok=True)
    logger.info(f"Saving images to {person_dir}")

    cap = cv2.VideoCapture(ip_camera)
    count = 0
    quality_threshold = 10000  # min face area in pixels

    while cap.isOpened() and count < num_images:
        ret, frame = cap.read()
        if not ret:
            logger.error("Camera read failed.")
            break

        results = model(frame, conf=0.5)
        frame_copy = frame.copy()

        if len(results) and len(results[0].boxes):
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                padding = 10
                x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
                x2, y2 = min(frame.shape[1], x2 + padding), min(frame.shape[0], y2 + padding)

                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue
                if face.shape[0] * face.shape[1] < quality_threshold:
                    continue

                Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB)).save(
                    os.path.join(person_dir, f"img_{count:03d}.png")
                )
                count += 1
                logger.info(f"Saved image {count}/{num_images} for {person_name}")
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("Face Capture", frame_copy)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            logger.info("User pressed 'q'. Exiting capture.")
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info(f"[SUCCESS] Saved {count} images for {person_name}")

if __name__ == "__main__":
    capture_faces()