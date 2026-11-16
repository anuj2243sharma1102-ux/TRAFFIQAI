"""Run the vehicle detector on a video, frame by frame.

Usage:
    python tests\\run_video_detection.py [video_path] [image_size] [confidence] [model_name]

Examples:
    python tests\\run_video_detection.py videos\\traffic.mp4
    python tests\\run_video_detection.py videos\\traffic.mp4 1280 0.25
    python tests\\run_video_detection.py videos\\cctv.mp4 960 0.35 yolov8s.pt

Press 'q' in the video window to stop.
"""

import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from detection.vehicle_detector import VehicleDetector

# ---- Read command-line options (all optional) ----
source = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "videos" / "traffic.mp4"
imgsz = int(sys.argv[2]) if len(sys.argv) > 2 else 960
conf = float(sys.argv[3]) if len(sys.argv) > 3 else 0.4
model_name = sys.argv[4] if len(sys.argv) > 4 else "yolov8n.pt"

if not source.exists():
    sys.exit(f"File not found: {source}")

cap = cv2.VideoCapture(str(source))
if not cap.isOpened():
    sys.exit("Could not open the video.")

# ---- Output video settings ----
MAX_WIDTH = 1280  # shrink big videos so the demo stays fast on CPU
in_fps = cap.get(cv2.CAP_PROP_FPS) or 25
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
if width > MAX_WIDTH:
    height = int(height * MAX_WIDTH / width)
    width = MAX_WIDTH

out_path = PROJECT_ROOT / "output" / "detected.mp4"
writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                         in_fps, (width, height))

# ---- Load the detector ----
detector = VehicleDetector(
    model_path=PROJECT_ROOT / "models" / model_name,
    confidence=conf,
    image_size=imgsz,
)
print(f"Model: {model_name} | image size: {imgsz} | confidence: {conf}")

counts = []  # vehicles found in each frame, used for the summary at the end
frame_count = 0
start_all = time.time()

# ---- Main loop: one frame at a time ----
while True:
    ok, frame = cap.read()
    if not ok:
        break  # end of video

    frame = cv2.resize(frame, (width, height))

    t0 = time.time()
    detections = detector.detect(frame)
    annotated = detector.draw(frame, detections)
    fps = 1.0 / max(time.time() - t0, 1e-6)

    counts.append(len(detections))

    cv2.putText(annotated, f"Vehicles: {len(detections)}   FPS: {fps:.1f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    writer.write(annotated)
    cv2.imshow("Vehicle Detection (press q to quit)", annotated)
    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()

# ---- Summary ----
elapsed = time.time() - start_all
print(f"Frames processed: {frame_count}")
print(f"Average speed: {frame_count / max(elapsed, 1e-6):.1f} FPS")
if counts:
    print(f"Vehicles per frame: average {sum(counts) / len(counts):.1f}, max {max(counts)}")
print("Saved:", out_path)