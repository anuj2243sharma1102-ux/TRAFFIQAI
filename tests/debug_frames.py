"""Save the picture of chosen moments with EVERY detection drawn (id, type, confidence).

Usage:
    python tests\\debug_frames.py 17.0 17.2 17.3 17.4

Green box = counted, orange box = not counted. A bike without any box was never detected.
Pictures go to output\\debug_frames (a strip of the road around the counting line).
"""

import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from detection.vehicle_counter import VehicleCounter
from detection.vehicle_detector import VehicleDetector

cv2.setNumThreads(1)

wanted = [float(a) for a in sys.argv[1:]]
if not wanted:
    sys.exit("Give the seconds, for example: python tests\\debug_frames.py 17.0 17.2 17.3")

folder = PROJECT_ROOT / "output" / "debug_frames"
folder.mkdir(parents=True, exist_ok=True)
for old in folder.glob("*.jpg"):
    old.unlink()

cap = cv2.VideoCapture(str(config.VIDEO_SOURCE))
if not cap.isOpened():
    sys.exit("Could not open the video.")
fps = cap.get(cv2.CAP_PROP_FPS) or 25
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
if width > config.MAX_WIDTH:
    height = int(height * config.MAX_WIDTH / width)
    width = config.MAX_WIDTH

detector = VehicleDetector(model_path=config.VEHICLE_MODEL, confidence=config.CONFIDENCE,
                           image_size=config.IMAGE_SIZE)
counter = VehicleCounter(line_ratio=config.LINE_RATIO)
line_y = int(height * config.LINE_RATIO)

targets = {round(t * fps): t for t in wanted}
last_index = max(targets)

index = 0
while index <= last_index:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.resize(frame, (width, height))
    detections = detector.track(frame)
    counter.update(detections, height, index / fps)
    if index in targets:
        picture = frame.copy()
        cv2.line(picture, (0, line_y), (width, line_y), (0, 255, 255), 2)
        for det in detections:
            vid = det["id"]
            x1, y1, x2, y2 = det["bbox"]
            counted = vid is not None and counter.alias.get(vid, vid) in counter.counted_ids
            color = (0, 200, 0) if counted else (0, 140, 255)
            cv2.rectangle(picture, (x1, y1), (x2, y2), color, 3)
            conf = det.get("conf", det.get("confidence"))
            text = f"id{vid if vid is not None else '-'} {det['label']}"
            if conf is not None:
                text += f" {conf:.2f}"
            cv2.putText(picture, text, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        strip = picture[max(0, line_y - 350): line_y + 350]
        name = f"{targets[index]:05.1f}s.jpg"
        cv2.imwrite(str(folder / name), strip)
        print(f"saved {name}: {len(detections)} detections", flush=True)
    index += 1
cap.release()
print("Pictures are in", folder)