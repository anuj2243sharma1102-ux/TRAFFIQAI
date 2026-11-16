"""Run the full pipeline with traffic violation detection.

Usage:
    python tests\\run_violations.py [video] [image_size] [confidence] [model] [redlight] [time_scale]

redlight   : 0 = red-light rule OFF (default), 1 = ON.
             Only meaningful with a REAL signal state; our signal is simulated.
time_scale : speed of the simulated signal clock (1 = real time).

Every violation is also saved as a picture in output\\violations.
Press 'q' in the video window to stop.
"""

import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from detection.signal_controller import SignalController
from detection.traffic_density import TrafficDensity
from detection.vehicle_counter import VehicleCounter
from detection.vehicle_detector import VehicleDetector
from detection.violation_detector import ViolationDetector

IGNORE_LEFT = 0.3    # e.g. 0.3 = ignore "stopped" in the left 30% of the frame (parked at the kerb)

source = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "videos" / "cctv.mp4"
imgsz = int(sys.argv[2]) if len(sys.argv) > 2 else 960
conf = float(sys.argv[3]) if len(sys.argv) > 3 else 0.35
model_name = sys.argv[4] if len(sys.argv) > 4 else "yolov8n.pt"
redlight = bool(int(sys.argv[5])) if len(sys.argv) > 5 else False
time_scale = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0

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

out_path = PROJECT_ROOT / "output" / "violations.mp4"
writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                         in_fps, (width, height))

pics_dir = PROJECT_ROOT / "output" / "violations"
pics_dir.mkdir(exist_ok=True)
for old in pics_dir.glob("*.jpg"):
    old.unlink()

detector = VehicleDetector(
    model_path=PROJECT_ROOT / "models" / model_name,
    confidence=conf,
    image_size=imgsz,
)
counter = VehicleCounter(line_ratio=0.4)
density = TrafficDensity()
signal = SignalController()
violations = ViolationDetector(flow="up", line_ratio=0.4, ignore_left=IGNORE_LEFT)
print(f"Model: {model_name} | red-light rule: {'ON' if redlight else 'OFF'}")

frame_count = 0
start_all = time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.resize(frame, (width, height))
    video_time = frame_count / in_fps

    detections = detector.track(frame)
    counter.update(detections, height, video_time)
    result = density.update(detections, frame.shape)
    info = signal.update(result["status"], video_time * time_scale)
    new = violations.update(detections, frame.shape, video_time,
                            density_status=result["status"],
                            signal_phase=info["phase"], check_red_light=redlight)

    annotated = detector.draw(frame, detections)
    counter.draw(annotated)
    density.draw(annotated, result)
    signal.draw(annotated, info)
    violations.draw(annotated, video_time)

    for ev in new:
        name = f"{ev['time']:05.1f}s_{ev['type']}_id{ev['id']}.jpg"
        cv2.imwrite(str(pics_dir / name), annotated)

    writer.write(annotated)
    cv2.imshow("Violation Detection (press q to quit)", annotated)
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
print("Violations found (video time | rule | vehicle id | type):")
if violations.events:
    for e in violations.events:
        print(f"  {e['time']:>5.1f}s | {e['type']:<9} | id #{e['id']:<5} | {e['label']}")
else:
    print("  none")
print("Summary:", violations.summary())
print("Pictures saved in:", pics_dir)
print("Saved:", out_path)