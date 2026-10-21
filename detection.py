

from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = Path(__file__).parent / "yolo26n.pt"
model = YOLO(str(MODEL_PATH))


def detect_objects(image):
    results = model(image, verbose=False)
    result = results[0]

    annotated_image = result.plot()
    counts = {}
    detections = []

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            label = model.names[class_id]

            counts[label] = counts.get(label, 0) + 1
            detections.append({
                "object": label,
                "confidence": round(confidence, 3),
            })

    return annotated_image, counts, detections
