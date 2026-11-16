"""Quick manual test: detect vehicles in one frame of a video (or an image).

Usage:
    python tests\\test_vehicle_detector.py <file> [image_size]
"""

import sys
import time
from collections import Counter
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))  # so "detection" can be imported

from detection.vehicle_detector import VehicleDetector

source = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "videos" / "traffic.mp4"
imgsz = int(sys.argv[2]) if len(sys.argv) > 2 else 960

if not source.exists():
    sys.exit(f"File not found: {source}")

if source.suffix.lower() in {".jpg", ".jpeg", ".png"}:
    frame = cv2.imread(str(source))
else:
    cap = cv2.VideoCapture(str(source))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)  # take a frame from the middle
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit("Could not read a frame from the video.")

print("Frame size:", frame.shape[1], "x", frame.shape[0])

detector = VehicleDetector(image_size=imgsz)

detector.detect(frame)  # warm-up run, the first call is always slower
start = time.time()
detections = detector.detect(frame)
print(f"Detection time: {time.time() - start:.2f} s  (imgsz={imgsz})")

print("Vehicles found:", len(detections))
print("By type:", dict(Counter(d["label"] for d in detections)))

out_path = PROJECT_ROOT / "output" / "test_detection.jpg"
cv2.imwrite(str(out_path), detector.draw(frame, detections))
print("Saved:", out_path)