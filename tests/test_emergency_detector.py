"""Test the emergency vehicle model on a photo or a video.

Usage:
    python tests\\test_emergency_detector.py [photo_or_video] [confidence] [start_seconds] [model_file]

Examples:
    python tests\\test_emergency_detector.py videos\\amb_fire.mp4 0.5 14 emergency2.pt
    python tests\\test_emergency_detector.py videos\\cctv.mp4 0.5 15 emergency2.pt
    python tests\\test_emergency_detector.py videos\\amb1.jpg 0.4

start_seconds : skip the first part of the video. Use it to test ONLY on the part
                that was NOT used for training (amb_fire: after 14 s, cctv: after 15 s).
Photo : prints every detection above the confidence.
Video : runs the detector with the voting rule, lists the confirmed vehicles and
        the raw detections. On normal traffic (cctv.mp4) any confirmed vehicle is a false alarm.
Press 'q' in the video window to stop.
"""

import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from detection.emergency_detector import EmergencyDetector

source = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "videos" / "cctv.mp4"
conf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
start_s = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
model_file = sys.argv[4] if len(sys.argv) > 4 else "emergency2.pt"

if not source.exists():
    sys.exit(f"File not found: {source}")

detector = EmergencyDetector(model_path=PROJECT_ROOT / "models" / model_file, conf=conf)
print(f"Model: {model_file} | classes: {detector.names} | confidence {conf} | "
      f"priority: {list(detector.priority_classes)}")

# ------------------------------------------------------------------ photo mode
if source.suffix.lower() in {".jpg", ".jpeg", ".png"}:
    frame = cv2.imread(str(source))
    if frame is None:
        sys.exit("Could not read the image.")
    result = detector.model(frame, conf=conf, imgsz=detector.imgsz, verbose=False)[0]
    print(f"Detections: {len(result.boxes)}")
    out = frame.copy()
    thickness = max(2, frame.shape[1] // 400)
    for box in result.boxes:
        name = detector.model.names[int(box.cls[0])]
        score = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        print(f"  {name:<13} score {score:.2f}  box {(x1, y1, x2, y2)}")
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 255), thickness)
        cv2.putText(out, f"{name} {score:.2f}", (x1, max(y1 - 8, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, max(0.6, frame.shape[1] / 1600),
                    (255, 0, 255), thickness)
    out_path = PROJECT_ROOT / "output" / "emergency_test.jpg"
    cv2.imwrite(str(out_path), out)
    print("Saved:", out_path)
    sys.exit()

# ------------------------------------------------------------------ video mode
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

start_frame = int(start_s * in_fps)
if start_frame > 0:
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
print(f"Testing from {start_s:g} s of the video onwards")

out_path = PROJECT_ROOT / "output" / "emergency.mp4"
writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                         in_fps, (width, height))

events = {}   # class name -> first time, last time, best score, frames
frame_index = start_frame
processed = 0
start_all = time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.resize(frame, (width, height))
    video_time = frame_index / in_fps

    confirmed = detector.update(frame)
    for det in confirmed:
        e = events.setdefault(det["label"], {"first": video_time, "best": 0.0, "frames": 0})
        e["last"] = video_time
        e["best"] = max(e["best"], det["score"])
        e["frames"] += 1

    annotated = frame.copy()
    detector.draw(annotated, confirmed)
    writer.write(annotated)
    cv2.imshow("Emergency Detection (press q to quit)", annotated)
    frame_index += 1
    processed += 1
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()

elapsed = time.time() - start_all
print(f"Frames processed: {processed}")
print(f"Average speed: {processed / max(elapsed, 1e-6):.1f} FPS")

if events:
    print("Confirmed (after the voting rule):")
    for name, e in events.items():
        print(f"  {name}: from {e['first']:.1f}s to {e['last']:.1f}s, "
              f"{e['frames']} frames, best score {e['best']:.2f}")
else:
    print("Nothing was confirmed.")

print("Raw detections (single checks, before the voting rule):")
if detector.raw_counts:
    for name, n in detector.raw_counts.items():
        print(f"  {name}: seen in {n} checks, best score {detector.raw_best[name]:.2f}")
else:
    print("  none")
print("Saved:", out_path)