"""Run the full pipeline with the adaptive traffic signal on a video.

Usage:
    python tests\\run_signal.py [video] [image_size] [confidence] [model]
                                [time_scale] [min_green] [max_green] [yellow] [red]
                                [emergency_window]

time_scale       : 1 = signal runs in real video time, 3 = signal clock runs 3x faster
                   (so a short test video shows several cycles).
emergency_window : simulated emergency vehicle as "start-end" in VIDEO seconds,
                   for example 8-12. You can also press 'e' in the video window to
                   switch a simulated emergency on and off.

Example:
    python tests\\run_signal.py videos\\cctv.mp4 960 0.35 yolov8n.pt 3 15 45 3 20 8-12

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

source = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "videos" / "cctv.mp4"
imgsz = int(sys.argv[2]) if len(sys.argv) > 2 else 960
conf = float(sys.argv[3]) if len(sys.argv) > 3 else 0.35
model_name = sys.argv[4] if len(sys.argv) > 4 else "yolov8n.pt"
time_scale = float(sys.argv[5]) if len(sys.argv) > 5 else 3.0
min_green = int(sys.argv[6]) if len(sys.argv) > 6 else 15
max_green = int(sys.argv[7]) if len(sys.argv) > 7 else 45
yellow = int(sys.argv[8]) if len(sys.argv) > 8 else 3
red = int(sys.argv[9]) if len(sys.argv) > 9 else 20

emergency_window = None
if len(sys.argv) > 10:
    try:
        start_s, end_s = sys.argv[10].split("-")
        emergency_window = (float(start_s), float(end_s))
    except ValueError:
        sys.exit("emergency_window must look like 8-12")

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

out_path = PROJECT_ROOT / "output" / "signal.mp4"
writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                         in_fps, (width, height))

detector = VehicleDetector(
    model_path=PROJECT_ROOT / "models" / model_name,
    confidence=conf,
    image_size=imgsz,
)
counter = VehicleCounter(line_ratio=0.4)
density = TrafficDensity()
signal = SignalController(min_green=min_green, max_green=max_green,
                          yellow=yellow, red=red)
print(f"Model: {model_name} | signal clock x{time_scale:g} | "
      f"green {min_green}-{max_green}s, yellow {yellow}s, red {red}s")
if emergency_window:
    print(f"Simulated emergency vehicle: video seconds {emergency_window[0]:g} to {emergency_window[1]:g}")

manual_emergency = False   # toggled with the 'e' key
frame_count = 0
start_all = time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.resize(frame, (width, height))

    video_time = frame_count / in_fps          # position in the video, seconds
    signal_time = video_time * time_scale      # signal clock (faster for demos)

    scheduled = emergency_window is not None and emergency_window[0] <= video_time < emergency_window[1]
    emergency = manual_emergency or scheduled

    detections = detector.track(frame)
    counter.update(detections, height, video_time)
    result = density.update(detections, frame.shape)
    info = signal.update(result["status"], signal_time, emergency=emergency)

    annotated = detector.draw(frame, detections)
    counter.draw(annotated)
    density.draw(annotated, result)
    signal.draw(annotated, info)
    if emergency:
        cv2.rectangle(annotated, (0, height - 110), (width, height - 72), (0, 0, 200), -1)
        cv2.putText(annotated, "EMERGENCY VEHICLE (simulated)", (8, height - 84),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    writer.write(annotated)
    cv2.imshow("Smart Traffic Signal (q = quit, e = emergency on/off)", annotated)
    frame_count += 1

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("e"):
        manual_emergency = not manual_emergency

cap.release()
writer.release()
cv2.destroyAllWindows()

elapsed = time.time() - start_all
print(f"Frames processed: {frame_count}")
print(f"Average speed: {frame_count / max(elapsed, 1e-6):.1f} FPS")
print(f"Vehicles crossed the line: {counter.stats()['total']}")
print("Signal log (signal time | video time | phase | length | density status when it started):")
for t, phase, length, status in signal.log:
    print(f"  {t:>6.1f}s | {t / time_scale:>5.1f}s | {phase:<6} | {length:>2}s | {status}")
print("Priority events (signal time | video time | note):")
if signal.events:
    for t, note in signal.events:
        print(f"  {t:>6.1f}s | {t / time_scale:>5.1f}s | {note}")
else:
    print("  none")
print("Saved:", out_path)