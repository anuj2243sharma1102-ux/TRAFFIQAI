"""Run detection + tracking + counting + traffic density on a video.

Usage:
    python tests\\run_density.py [video] [image_size] [confidence] [model]
                                 [low_max] [high_min] [roi_top] [roi_bottom]

Example:
    python tests\\run_density.py videos\\cctv.mp4 960 0.35 yolov8n.pt 0.08 0.18 0.15 0.55

Press 'q' in the video window to stop.
"""

import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from detection.traffic_density import TrafficDensity
from detection.vehicle_counter import VehicleCounter
from detection.vehicle_detector import VehicleDetector

source = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "videos" / "cctv.mp4"
imgsz = int(sys.argv[2]) if len(sys.argv) > 2 else 960
conf = float(sys.argv[3]) if len(sys.argv) > 3 else 0.35
model_name = sys.argv[4] if len(sys.argv) > 4 else "yolov8n.pt"
low_max = float(sys.argv[5]) if len(sys.argv) > 5 else 0.12
high_min = float(sys.argv[6]) if len(sys.argv) > 6 else 0.20
roi_top = float(sys.argv[7]) if len(sys.argv) > 7 else 0.15
roi_bottom = float(sys.argv[8]) if len(sys.argv) > 8 else 0.55

if not source.exists():
    sys.exit(f"File not found: {source}")

cap = cv2.VideoCapture(str(source))
if not cap.isOpened():
    sys.exit("Could not open the video.")

MAX_WIDTH = 1280
in_fps = cap.get(cv2.CAP_PROP_FPS) or 25
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
if width > MAX_WIDTH:
    height = int(height * MAX_WIDTH / width)
    width = MAX_WIDTH

out_path = PROJECT_ROOT / "output" / "density.mp4"
writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                         in_fps, (width, height))

detector = VehicleDetector(
    model_path=PROJECT_ROOT / "models" / model_name,
    confidence=conf,
    image_size=imgsz,
)
counter = VehicleCounter(line_ratio=0.4)
roi = [(0.0, roi_top), (1.0, roi_top), (1.0, roi_bottom), (0.0, roi_bottom)]
density = TrafficDensity(roi=roi, low_max=low_max, high_min=high_min)
print(f"Model: {model_name} | size: {imgsz} | conf: {conf} | "
      f"thresholds: Low < {low_max:.0%} <= Medium < {high_min:.0%} <= High | "
      f"ROI: {roi_top:.0%}-{roi_bottom:.0%} of frame height")

raw_occupancy = []                 # occupancy of every frame
status_frames = Counter()          # how many frames in each status
status_changes = 0                 # how often the status flipped (flicker check)
last_status = None
windows = defaultdict(list)        # 5-second window -> occupancy values
frame_count = 0
start_all = time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.resize(frame, (width, height))

    detections = detector.track(frame)
    video_time = frame_count / in_fps
    counter.update(detections, height, video_time)
    result = density.update(detections, frame.shape)

    raw_occupancy.append(result["occupancy"])
    status_frames[result["status"]] += 1
    if last_status is not None and result["status"] != last_status:
        status_changes += 1
    last_status = result["status"]
    windows[int(video_time // 5)].append(result["smoothed"])

    annotated = detector.draw(frame, detections)
    counter.draw(annotated)
    density.draw(annotated, result)

    writer.write(annotated)
    cv2.imshow("Traffic Density (press q to quit)", annotated)
    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()

elapsed = time.time() - start_all
print(f"Frames processed: {frame_count}")
print(f"Average speed: {frame_count / max(elapsed, 1e-6):.1f} FPS")
print(f"Vehicles crossed the line: {counter.stats()['total']}")

if raw_occupancy:
    p = np.percentile(raw_occupancy, [10, 25, 50, 75, 90])
    print("Occupancy (share of ROI covered by vehicles):")
    print(f"  min {min(raw_occupancy):.1%} | p10 {p[0]:.1%} | p25 {p[1]:.1%} | "
          f"median {p[2]:.1%} | p75 {p[3]:.1%} | p90 {p[4]:.1%} | max {max(raw_occupancy):.1%}")
    print("Frames per status:",
          {k: f"{v / frame_count:.0%}" for k, v in status_frames.items()})
    print(f"Status changes during the video: {status_changes}")
    print("Smoothed occupancy per 5 s window:")
    for w in sorted(windows):
        values = windows[w]
        print(f"  {w * 5:>3}-{w * 5 + 5:<3} s : {sum(values) / len(values):.1%}")
print("Saved:", out_path)