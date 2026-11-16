"""Run detection + tracking + counting on a video.

Usage:
    python tests\\run_counting.py [video] [image_size] [confidence] [model] [line_ratio]

Example:
    python tests\\run_counting.py videos\\cctv.mp4 960 0.35 yolov8n.pt 0.4

Press 'q' in the video window to stop.
"""

import sys
import time
from collections import Counter
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from detection.vehicle_counter import VehicleCounter
from detection.vehicle_detector import VehicleDetector

source = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "videos" / "cctv.mp4"
imgsz = int(sys.argv[2]) if len(sys.argv) > 2 else 960
conf = float(sys.argv[3]) if len(sys.argv) > 3 else 0.35
model_name = sys.argv[4] if len(sys.argv) > 4 else "yolov8n.pt"
line_ratio = float(sys.argv[5]) if len(sys.argv) > 5 else 0.4

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

out_path = PROJECT_ROOT / "output" / "counted.mp4"
writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                         in_fps, (width, height))

detector = VehicleDetector(
    model_path=PROJECT_ROOT / "models" / model_name,
    confidence=conf,
    image_size=imgsz,
)
counter = VehicleCounter(line_ratio=line_ratio)
print(f"Model: {model_name} | size: {imgsz} | conf: {conf} | line at {line_ratio:.0%}")

frame_count = 0
start_all = time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.resize(frame, (width, height))

    detections = detector.track(frame)
    video_time = frame_count / in_fps          # position in the video, in seconds
    counter.update(detections, height, video_time)

    annotated = detector.draw(frame, detections)
    counter.draw(annotated)

    writer.write(annotated)
    cv2.imshow("Vehicle Counting (press q to quit)", annotated)
    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()

elapsed = time.time() - start_all
stats = counter.stats()
print(f"Frames processed: {frame_count}")
print(f"Average speed: {frame_count / max(elapsed, 1e-6):.1f} FPS")
print(f"Total vehicles crossed the line: {stats['total']} "
      f"(down {stats['down']}, up {stats['up']})")
print("By type:", stats["by_type"])

# Crossings per 5-second window of the video, to compare with a hand count
windows = Counter(int(t // 5) for t, _, _, _ in counter.events)
print("Crossings per 5 s window of the video:")
for w in range(int(frame_count / in_fps // 5) + 1):
    print(f"  {w * 5:>3}-{w * 5 + 5:<3} s : {windows.get(w, 0)}")
print("Saved:", out_path)