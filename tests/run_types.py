"""Run tracking + our vehicle type classifier on a video and compare the vehicle
types counted with YOLO alone against the types after our own classifier.

Usage:
    python tests\\run_types.py [video] [min_conf]

Example:
    python tests\\run_types.py videos\\cctv.mp4 0.6
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
from detection.vehicle_type_classifier import VehicleTypeClassifier

source = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "videos" / "cctv.mp4"
min_conf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.6

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

out_path = PROJECT_ROOT / "output" / "types.mp4"
writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                         in_fps, (width, height))

detector = VehicleDetector(model_path=PROJECT_ROOT / "models" / "yolov8n.pt",
                           confidence=0.35, image_size=960)
classifier = VehicleTypeClassifier(min_conf=min_conf)
print(f"Classes: {classifier.classes} | confidence needed: {min_conf}")

counter_yolo = VehicleCounter(line_ratio=0.4)    # counts with YOLO's labels
counter_typed = VehicleCounter(line_ratio=0.4)   # counts with our labels
final_labels = {}                                # vehicle id -> (YOLO label, our label)
frame_count = 0
start_all = time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.resize(frame, (width, height))
    video_time = frame_count / in_fps

    detections = detector.track(frame)
    typed = classifier.update(frame, detections)
    counter_yolo.update(detections, height, video_time)
    counter_typed.update(typed, height, video_time)
    for det in typed:
        if det["id"] is not None:
            final_labels[det["id"]] = (det["yolo_label"], det["label"])

    annotated = detector.draw(frame, typed)
    counter_typed.draw(annotated)
    writer.write(annotated)
    cv2.imshow("Vehicle types (press q to quit)", annotated)
    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()

elapsed = time.time() - start_all
print(f"Frames processed: {frame_count}")
print(f"Average speed: {frame_count / max(elapsed, 1e-6):.1f} FPS")
print(f"Vehicles crossed the line: YOLO {counter_yolo.stats()['total']} | "
      f"with classifier {counter_typed.stats()['total']}")
print("By type, YOLO labels only :", counter_yolo.stats()["by_type"])
print("By type, with classifier  :", counter_typed.stats()["by_type"])

changes = Counter((y, t) for y, t in final_labels.values() if y != t)
print("Labels that were changed (YOLO label -> our label : number of vehicles):")
if changes:
    for (y, t), n in changes.most_common(12):
        print(f"  {y:<10} -> {t:<10} : {n}")
else:
    print("  none")
print("Saved:", out_path)