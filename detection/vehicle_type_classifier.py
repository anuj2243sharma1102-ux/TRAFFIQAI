"""Vehicle type classifier: car, taxi, auto, tempo, bus, truck, motorcycle ...

YOLO only knows car / motorcycle / bus / truck. This module looks at the crop of
every tracked vehicle with a small classifier we trained on our own crops
(tools\\train_vehicle_type.py) and gives it a better name.

To avoid flicker the answers of the last few checks are averaged per vehicle ID.
If the classifier is not sure enough (below min_conf), YOLO's own label is kept.
"""

from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "vehicle_type.pt"

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class VehicleTypeClassifier:
    def __init__(self, model_path=DEFAULT_MODEL, min_conf=0.6, every_n=5, window=8, min_box=40):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Vehicle type model not found: {model_path}")

        saved = torch.load(model_path, weights_only=True)
        self.classes = saved["classes"]
        self.weight, self.bias = saved["weight"], saved["bias"]
        self.mean, self.std = saved["mean"], saved["std"]
        self.net = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT).eval()

        self.min_conf = min_conf    # below this confidence we keep YOLO's label
        self.every_n = every_n      # re-check each vehicle every N frames
        self.window = window        # how many recent checks are averaged
        self.min_box = min_box      # ignore vehicles smaller than this (pixels)

        self.frame_idx = 0
        self.last_checked = {}      # vehicle id -> frame of the last check
        self.history = {}           # vehicle id -> recent probability vectors

    def _prepare(self, crop):
        rgb = cv2.cvtColor(cv2.resize(crop, (224, 224)), cv2.COLOR_BGR2RGB)
        x = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
        return torch.from_numpy(x.transpose(2, 0, 1))

    def _probabilities(self, crops):
        batch = torch.stack([self._prepare(c) for c in crops])
        with torch.no_grad():
            features = self.net.avgpool(self.net.features(batch)).flatten(1)
            logits = torch.nn.functional.linear((features - self.mean) / self.std,
                                                self.weight, self.bias)
        return torch.softmax(logits, dim=1)

    def update(self, frame, detections):
        """Return a copy of the detections with better labels (the YOLO label is kept
        as "yolo_label")."""
        self.frame_idx += 1

        due, crops = [], []
        for det in detections:
            vid = det["id"]
            if vid is None:
                continue
            x1, y1, x2, y2 = det["bbox"]
            if (x2 - x1) < self.min_box or (y2 - y1) < self.min_box:
                continue
            if self.frame_idx - self.last_checked.get(vid, -10**9) < self.every_n:
                continue
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size:
                due.append(vid)
                crops.append(crop)

        if crops:
            for vid, probs in zip(due, self._probabilities(crops)):
                self.last_checked[vid] = self.frame_idx
                self.history.setdefault(vid, deque(maxlen=self.window)).append(probs)

        typed = []
        for det in detections:
            new = dict(det)
            new["yolo_label"] = det["label"]
            checks = self.history.get(det["id"])
            if checks:
                average = torch.stack(list(checks)).mean(0)
                confidence, index = average.max(0)
                if float(confidence) >= self.min_conf:
                    new["label"] = self.classes[int(index)]
                    new["type_conf"] = round(float(confidence), 2)
            typed.append(new)
        return typed