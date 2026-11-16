"""Save one picture for every vehicle the program counts, to check the count by eye.

Usage:
    python tests\\audit_counting.py 20 30

Plays the video from the start up to the last second given (about 3 to 4 minutes) and saves
a picture for every count between the two seconds into output\\count_audit:
  - the whole frame at the moment of counting,
  - the counted vehicle in a RED box, the counting line in yellow,
  - a number, time, ID and type written on the picture.
A contact_sheet.jpg with all pictures side by side is made as well.
Look at the pictures in order: the same vehicle in two neighbouring pictures = double count.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from detection.vehicle_counter import VehicleCounter
from detection.vehicle_detector import VehicleDetector

cv2.setNumThreads(1)

first = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0
last = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0

folder = PROJECT_ROOT / "output" / "count_audit"
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

print(f"Playing the video up to {last:.0f} s, saving a picture for every count from "
      f"{first:.0f} s to {last:.0f} s ...", flush=True)
saved = 0
index = 0
while True:
    ok, frame = cap.read()
    t = index / fps
    if not ok or t > last:
        break
    frame = cv2.resize(frame, (width, height))
    before = len(counter.events)
    detections = detector.track(frame)
    counter.update(detections, height, t)

    for event_time, vid, label, direction in counter.events[before:]:
        if not (first <= event_time < last):
            continue
        saved += 1
        picture = frame.copy()
        cv2.line(picture, (0, line_y), (width, line_y), (0, 255, 255), 4)
        det = next((d for d in detections if d["id"] == vid), None)
        if det:
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(picture, (x1, y1), (x2, y2), (0, 0, 255), 8)
        text = f"#{saved}  {event_time:.1f}s  id{vid} {label}"
        cv2.rectangle(picture, (0, 0), (width, 90), (0, 0, 0), -1)
        cv2.putText(picture, text, (14, 66), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 5)
        small = cv2.resize(picture, (540, int(height * 540 / width)))
        cv2.imwrite(str(folder / f"{saved:02d}_{event_time:05.1f}s_id{vid}_{label}.jpg"), small)
    index += 1
    if index % 150 == 0:
        print(f"  {t:5.1f} s done, {counter.stats()['total']} counted so far", flush=True)
cap.release()

pictures = sorted(folder.glob("[0-9][0-9]_*.jpg"))
print(f"\n{saved} counts between {first:.0f} s and {last:.0f} s, {len(pictures)} pictures saved in {folder}")

if pictures:
    thumbs = []
    for path in pictures:
        image = cv2.imread(str(path))
        thumbs.append(cv2.resize(image, (300, int(image.shape[0] * 300 / image.shape[1]))))
    columns = 7
    blank = np.zeros_like(thumbs[0])
    while len(thumbs) % columns:
        thumbs.append(blank)
    rows = [np.hstack(thumbs[i:i + columns]) for i in range(0, len(thumbs), columns)]
    sheet_path = folder / "contact_sheet.jpg"
    cv2.imwrite(str(sheet_path), np.vstack(rows))
    print("Contact sheet:", sheet_path)