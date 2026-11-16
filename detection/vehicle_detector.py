"""Vehicle detection and tracking using a YOLO model.

detect(frame): vehicles in ONE frame, no memory between frames.
track(frame) : same, but each vehicle also gets an "id" that stays the same
               from frame to frame (used for counting).
"""

from pathlib import Path

import cv2
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "yolov8n.pt"

# COCO class IDs that YOLO already knows, mapped to our vehicle names
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Box colours in BGR (OpenCV uses Blue-Green-Red, not RGB)
COLORS = {
    "car": (0, 200, 0),
    "motorcycle": (0, 165, 255),
    "bus": (255, 120, 0),
    "truck": (0, 0, 220),
    "auto": (0, 255, 255),      # yellow
    "taxi": (147, 20, 255),     # pink
    "tempo": (200, 0, 150),     # purple
}


class VehicleDetector:
    def __init__(self, model_path=DEFAULT_MODEL, confidence=0.4, image_size=960,
                 tracker="bytetrack.yaml"):
        # If the weights file is missing, Ultralytics downloads it
        self.model = YOLO(str(model_path))
        self.confidence = confidence
        self.image_size = image_size
        self.tracker = str(tracker)       # tracker settings: "bytetrack.yaml" or a path to a yaml file

    def _parse(self, results):
        """Convert YOLO results into a simple list of dictionaries."""
        detections = []
        for box in results.boxes:
            class_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            track_id = int(box.id[0]) if box.id is not None else None
            detections.append(
                {
                    "id": track_id,
                    "label": VEHICLE_CLASSES[class_id],
                    "confidence": round(float(box.conf[0]), 2),
                    "bbox": (x1, y1, x2, y2),
                }
            )
        return detections

    def detect(self, frame):
        """Detect vehicles in one frame (no tracking)."""
        results = self.model(
            frame,
            conf=self.confidence,
            imgsz=self.image_size,
            classes=list(VEHICLE_CLASSES),
            agnostic_nms=True,          # merge overlapping boxes of different classes
            verbose=False,
        )[0]
        return self._parse(results)

    def track(self, frame):
        """Detect vehicles and give each one an ID that persists across frames."""
        results = self.model.track(
            frame,
            persist=True,               # remember vehicles from earlier frames
            tracker=self.tracker,
            conf=self.confidence,
            imgsz=self.image_size,
            classes=list(VEHICLE_CLASSES),
            agnostic_nms=True,          # merge overlapping boxes of different classes
            verbose=False,
        )[0]
        return self._parse(results)

    def draw(self, frame, detections):
        """Draw a box and label on a copy of the frame."""
        output = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            color = COLORS.get(det["label"], (200, 200, 200))
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            if det["id"] is not None:
                text = f'#{det["id"]} {det["label"]}'
            else:
                text = f'{det["label"]} {det["confidence"]:.2f}'
            cv2.putText(output, text, (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return output