"""Cut vehicle crops out of a video and sort them into folders by YOLO's label.

You then fix the wrong ones by hand (drag and drop in File Explorer), for example
move the auto-rickshaws from "truck" into a new folder "auto".

Usage:
    python tools\\harvest_crops.py [video] [max_per_class] [model]

Example:
    python tools\\harvest_crops.py videos\\cctv.mp4 120

Crops go to datasets\\vehicle_crops\\<label>\\ . The file name contains the video time
(for example cctv_023.4s_id12.jpg), which we use later to keep a test part apart.
"""

import sys
from collections import defaultdict
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from detection.vehicle_detector import VehicleDetector

source = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "videos" / "cctv.mp4"
max_per_class = int(sys.argv[2]) if len(sys.argv) > 2 else 120
model_name = sys.argv[3] if len(sys.argv) > 3 else "yolov8n.pt"

MIN_SIDE = 48      # skip tiny crops (far away vehicles)
PER_ID = 2         # crops kept per tracked vehicle
GAP_S = 1.0        # seconds between two crops of the same vehicle

out_root = PROJECT_ROOT / "datasets" / "vehicle_crops"
if out_root.exists():
    sys.exit(f"{out_root} already exists. It may contain your sorted crops, so I will not "
             f"touch it. Rename or delete it first if you really want to start again.")
if not source.exists():
    sys.exit(f"File not found: {source}")

cap = cv2.VideoCapture(str(source))
if not cap.isOpened():
    sys.exit("Could not open the video.")
fps = cap.get(cv2.CAP_PROP_FPS) or 25

detector = VehicleDetector(model_path=PROJECT_ROOT / "models" / model_name,
                           confidence=0.35, image_size=960)

candidates = defaultdict(list)   # YOLO label -> [(time, id, crop)]
taken = defaultdict(int)         # id -> how many crops we kept
last_time = {}                   # id -> time of the last crop
frame_idx = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    t = frame_idx / fps
    frame_idx += 1
    h, w = frame.shape[:2]

    for det in detector.track(frame):
        vid = det["id"]
        if vid is None or taken[vid] >= PER_ID:
            continue
        if t - last_time.get(vid, -1e9) < GAP_S:
            continue
        x1, y1, x2, y2 = det["bbox"]
        if x2 - x1 < MIN_SIDE or y2 - y1 < MIN_SIDE:
            continue
        if x1 <= 2 or y1 <= 2 or x2 >= w - 2 or y2 >= h - 2:   # cut off by the frame edge
            continue
        candidates[det["label"]].append((t, vid, frame[y1:y2, x1:x2].copy()))
        taken[vid] += 1
        last_time[vid] = t

cap.release()

for label, items in candidates.items():
    items.sort(key=lambda item: item[0])
    if len(items) > max_per_class:     # keep an even spread over the video
        items = [items[int(i * len(items) / max_per_class)] for i in range(max_per_class)]
    folder = out_root / label
    folder.mkdir(parents=True, exist_ok=True)
    for t, vid, crop in items:
        cv2.imwrite(str(folder / f"{source.stem}_{t:05.1f}s_id{vid}.jpg"), crop)
    print(f"{label:<11}: {len(items)} crops")

print("Saved in:", out_root)