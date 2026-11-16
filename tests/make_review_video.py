"""Make a video that shows what the program counts, to look for MISSED vehicles.

Usage:
    python tests\\make_review_video.py

A green box with a number appears on a vehicle at the moment the program counts it and
stays for one second. Watch the video: a vehicle that crosses the yellow line from the
bottom to the top WITHOUT a green box was missed by the program.
Output: output\\review_video.mp4   (takes about 4 minutes)
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
HOLD_SECONDS = 1.0       # how long the green box stays on the screen
OUT_WIDTH = 540
FONT = cv2.FONT_HERSHEY_SIMPLEX

cap = cv2.VideoCapture(str(config.VIDEO_SOURCE))
if not cap.isOpened():
    sys.exit("Could not open the video.")
fps = cap.get(cv2.CAP_PROP_FPS) or 25
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
if width > config.MAX_WIDTH:
    height = int(height * config.MAX_WIDTH / width)
    width = config.MAX_WIDTH
out_h = int(height * OUT_WIDTH / width)
scale = OUT_WIDTH / width
line_y = int(out_h * config.LINE_RATIO)

out_path = PROJECT_ROOT / "output" / "review_video.mp4"
out_path.parent.mkdir(exist_ok=True)
writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (OUT_WIDTH, out_h))

detector = VehicleDetector(model_path=config.VEHICLE_MODEL, confidence=config.CONFIDENCE,
                           image_size=config.IMAGE_SIZE)
counter = VehicleCounter(line_ratio=config.LINE_RATIO)

last_box = {}     # vehicle id -> last known box
shown = []        # (time it was counted, vehicle id, number)
print("Making the review video, please wait (about 4 minutes) ...", flush=True)

index = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    t = index / fps
    frame = cv2.resize(frame, (width, height))

    before = len(counter.events)
    detections = detector.track(frame)
    counter.update(detections, height, t)
    for d in detections:
        if d["id"] is not None:
            last_box[d["id"]] = d["bbox"]
    for number, event in enumerate(counter.events[before:], start=before + 1):
        shown.append((t, event[1], number))
    shown = [s for s in shown if t - s[0] <= HOLD_SECONDS]

    picture = cv2.resize(frame, (OUT_WIDTH, out_h))
    cv2.line(picture, (0, line_y), (OUT_WIDTH, line_y), (0, 255, 255), 2)
    for _, vid, number in shown:
        box = last_box.get(vid)
        if box:
            x1, y1, x2, y2 = [int(v * scale) for v in box]
            cv2.rectangle(picture, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(picture, str(number), (x1, max(y1 - 6, 50)), FONT, 0.9, (0, 255, 0), 2)
    cv2.rectangle(picture, (0, 0), (OUT_WIDTH, 34), (0, 0, 0), -1)
    cv2.putText(picture, f"{t:5.1f} s    counted: {counter.stats()['total']}", (8, 24),
                FONT, 0.7, (255, 255, 255), 2)
    writer.write(picture)

    index += 1
    if index % 150 == 0:
        print(f"  {t:5.1f} s done", flush=True)

cap.release()
writer.release()
print("Saved:", out_path)
print("Total counted by the program:", counter.stats()["total"])