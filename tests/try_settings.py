"""Try other detector settings and compare how many vehicles are counted.

Usage:
    python tests\\try_settings.py 960:0.25 1280:0.25 1280:0.20:sensitive

Every setting is  image_size:confidence  or  image_size:confidence:sensitive
("sensitive" = a tracker that also starts tracks for weaker detections).
Each setting plays the whole video once and only counts (4 to 7 minutes each).
The counts of every setting are saved in output\\events_<setting>.json.

More vehicles counted is NOT automatically better: extra counts can be double counts.
The pictures (audit) decide.
"""

import json
import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from detection.vehicle_counter import VehicleCounter
from detection.vehicle_detector import VehicleDetector

cv2.setNumThreads(1)
SENSITIVE = str(PROJECT_ROOT / "detection" / "bytetrack_sensitive.yaml")

settings = []
for arg in sys.argv[1:]:
    parts = arg.split(":")
    try:
        size, conf = int(parts[0]), float(parts[1])
    except (ValueError, IndexError):
        sys.exit(f"Cannot read '{arg}'. Use image_size:confidence, for example 1280:0.25")
    tracker = SENSITIVE if len(parts) > 2 and parts[2] == "sensitive" else "bytetrack.yaml"
    settings.append((arg, size, conf, tracker))
if not settings:
    sys.exit(__doc__)


def play(size, conf, tracker):
    cap = cv2.VideoCapture(str(config.VIDEO_SOURCE))
    if not cap.isOpened():
        sys.exit("Could not open the video.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width > config.MAX_WIDTH:
        height = int(height * config.MAX_WIDTH / width)
        width = config.MAX_WIDTH
    detector = VehicleDetector(model_path=config.VEHICLE_MODEL, confidence=conf,
                               image_size=size, tracker=tracker)
    counter = VehicleCounter(line_ratio=config.LINE_RATIO)
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (width, height))
        counter.update(detector.track(frame), height, index / fps)
        index += 1
    cap.release()
    return counter.stats(), [list(e) for e in counter.events]


rows = []
for arg, size, conf, tracker in settings:
    print(f"running {arg} (please wait, several minutes) ...", flush=True)
    start = time.time()
    stats, events = play(size, conf, tracker)
    seconds = time.time() - start
    rows.append((arg, seconds, stats))
    out = PROJECT_ROOT / "output" / f"events_{arg.replace(':', '_')}.json"
    out.write_text(json.dumps(events))
    print(f"  done in {seconds:.0f} s, {stats['total']} counted", flush=True)

print()
print("setting            time (s)  counted  motorcycle  car  truck  bus")
print("960:0.35 (before)       228       54          18   18     12    6   <- earlier run")
for arg, seconds, stats in rows:
    t = stats["by_type"]
    print(f"{arg:<18} {seconds:>8.0f} {stats['total']:>8} {t.get('motorcycle', 0):>11} "
          f"{t.get('car', 0):>4} {t.get('truck', 0):>6} {t.get('bus', 0):>4}")
print("\n(types are YOLO's own labels; auto-rickshaws appear as car or truck, tempos as bus)")