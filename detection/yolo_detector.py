from pathlib import Path
from ultralytics import YOLO

class YOLODetector:

    def __init__(self):
        weights = Path(__file__).parent / "yolov8n.pt"
        self.model = YOLO(str(weights))

    def detect(self, image):
        results = self.model(image)

        detections = []

        r = results[0]

        for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
            detections.append({ "label":      r.names[int(cls)],
                                "confidence": float(conf),
                                "bbox":       box.tolist()        })

        return detections
