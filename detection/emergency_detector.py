"""Emergency vehicle detection with our own trained YOLO model.

The model (models/emergency2.pt) was trained on Google Colab with the classes
Ambulance and Fire Engine (we label our own frames in Roboflow).

A single detection is not trusted: a class must be seen in at least
`min_hits` of the last `window` checks before it counts as confirmed.
Class names are compared in lowercase, so "Ambulance" and "ambulance" are the same.
"""

from collections import Counter, deque
from pathlib import Path

import cv2
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "emergency2.pt"

ALERT_COLOR = (255, 0, 255)   # magenta (BGR): priority vehicles
INFO_COLOR = (255, 255, 0)    # cyan (BGR): other classes


class EmergencyDetector:
    def __init__(self, model_path=DEFAULT_MODEL, conf=0.5, imgsz=640, every_n=2,
                 window=6, min_hits=3,
                 priority_classes=("ambulance", "fire engine", "fire brigade", "fire truck")):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Emergency model not found: {model_path}")

        self.model = YOLO(str(model_path))
        self.names = [str(n) for n in self.model.names.values()]   # e.g. ["Ambulance", "Fire Engine"]
        self.conf = conf                    # minimum confidence of one detection
        self.imgsz = imgsz                  # size the model looks at (it was trained at 640)
        self.every_n = every_n              # run the model on every N-th frame (saves time)
        self.min_hits = min_hits            # checks (out of `window`) that must agree
        self.priority_classes = tuple(p.lower() for p in priority_classes)

        self.frame_idx = 0
        self.hits = {name: deque(maxlen=window) for name in self.names}
        self.latest = []                    # detections of the most recent check
        self.raw_counts = Counter()         # checks in which each class was seen (for reports)
        self.raw_best = {}                  # best confidence ever seen per class

    def update(self, frame):
        """Call once per frame. Returns the confirmed emergency detections."""
        self.frame_idx += 1

        if (self.frame_idx - 1) % self.every_n == 0:
            result = self.model(frame, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
            self.latest = []
            seen = set()
            for box in result.boxes:
                name = str(self.model.names[int(box.cls[0])])
                score = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                self.latest.append({"label": name, "score": round(score, 2),
                                    "bbox": (x1, y1, x2, y2)})
                seen.add(name)
                self.raw_best[name] = max(self.raw_best.get(name, 0.0), score)
            for name in self.names:
                self.hits[name].append(name in seen)
            for name in seen:
                self.raw_counts[name] += 1

        confirmed = []
        for det in self.latest:
            if sum(self.hits[det["label"]]) >= self.min_hits:
                confirmed.append({**det,
                                  "priority": det["label"].lower() in self.priority_classes})
        return confirmed

    def draw(self, frame, confirmed):
        """Draw confirmed detections; show a banner if a priority vehicle is present."""
        for det in confirmed:
            x1, y1, x2, y2 = det["bbox"]
            color = ALERT_COLOR if det["priority"] else INFO_COLOR
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4 if det["priority"] else 2)
            cv2.putText(frame, f"{det['label']} {det['score']:.2f}", (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if any(det["priority"] for det in confirmed):
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, h - 110), (w, h - 72), (0, 0, 200), -1)
            cv2.putText(frame, "EMERGENCY VEHICLE DETECTED", (8, h - 84),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return frame