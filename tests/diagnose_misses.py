"""Find out WHY a vehicle near the line was not counted.

Usage:
    python tests\\diagnose_misses.py 0.4 0.8 7.6 14 17.2

Give the seconds where you saw a vehicle cross the line WITHOUT a green box.
For every second, it lists the tracks near the line that were NOT counted.
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

times = [float(a) for a in sys.argv[1:]]
if not times:
    sys.exit("Give the seconds, for example: python tests\\diagnose_misses.py 0.4 0.8 7.6")
MARGIN = 0.6                      # look this many seconds before and after
last = max(times) + MARGIN + 0.5

cap = cv2.VideoCapture(str(config.VIDEO_SOURCE))
if not cap.isOpened():
    sys.exit("Could not open the video.")
fps = cap.get(cv2.CAP_PROP_FPS) or 25
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
if width > config.MAX_WIDTH:
    height = int(height * config.MAX_WIDTH / width)
    width = config.MAX_WIDTH

detector = VehicleDetector(model_path=config.VEHICLE_MODEL, confidence=config.CONFIDENCE,
                           image_size=config.IMAGE_SIZE)
counter = VehicleCounter(line_ratio=config.LINE_RATIO)
line_y = int(height * config.LINE_RATIO)
band = int(height * 0.08)         # "near the line" = 8% of the frame height above/below

tracks = {}
index = 0
print(f"Playing the video up to {last:.1f} s ...", flush=True)
while True:
    ok, frame = cap.read()
    t = index / fps
    if not ok or t > last:
        break
    frame = cv2.resize(frame, (width, height))
    detections = detector.track(frame)
    counter.update(detections, height, t)
    for det in detections:
        vid = det["id"]
        if vid is None:
            continue
        x1, y1, x2, y2 = det["bbox"]
        cy = (y1 + y2) // 2
        cx = (x1 + x2) // 2
        conf = det.get("conf", det.get("confidence", 0.0))
        info = tracks.get(vid)
        if info is None:
            tracks[vid] = {"label": det["label"], "first_t": t, "first_cy": cy,
                           "last_t": t, "last_cy": cy, "min_cy": cy, "max_cy": cy,
                           "cx": cx, "frames": 1}
        else:
            info["last_t"] = t
            info["last_cy"] = cy
            info["min_cy"] = min(info["min_cy"], cy)
            info["max_cy"] = max(info["max_cy"], cy)
            info["cx"] = cx
            info["frames"] += 1
    index += 1
cap.release()

print(f"\nLine is at y = {line_y} (frame height {height}). Frames per second: {fps:.0f}")
for target in times:
    lo, hi = target - MARGIN, target + MARGIN
    print(f"\n=== around {target} s  (window {lo:.1f}-{hi:.1f} s) ===")
    found = 0
    for vid, info in sorted(tracks.items(), key=lambda kv: kv[1]["first_t"]):
        if info["last_t"] < lo or info["first_t"] > hi:
            continue
        if not (info["min_cy"] <= line_y + band and info["max_cy"] >= line_y - band):
            continue                    # never came near the line
        if counter.alias.get(vid, vid) in counter.counted_ids:
            continue                    # was counted, not a miss
        found += 1
        if info["first_cy"] < line_y - band // 2:
            start = "BORN ABOVE the line (track started after the line)"
        elif info["first_cy"] > line_y + band // 2:
            start = "born below the line"
        else:
            start = "born AT the line"
        print(f"  id{vid} {info['label']:<10} x={info['cx']:>4}  seen {info['first_t']:.1f}-{info['last_t']:.1f} s, "
              f"{info['frames']} frames, y {info['first_cy']} -> {info['last_cy']}  | {start}")
    if not found:
        print("  no uncounted track near the line -> YOLO probably never detected it (detection miss)")