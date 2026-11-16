"""Quick check of TrafficDensity with fake detections (no video needed)."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from detection.traffic_density import TrafficDensity

FULL = [(0, 0), (1, 0), (1, 1), (0, 1)]   # ROI = whole frame
shape = (100, 100, 3)                      # a 100 x 100 frame


def box(x1, y1, x2, y2):
    return {"id": 1, "label": "car", "confidence": 0.9, "bbox": (x1, y1, x2, y2)}


# ---- Part 1: the maths (status changes immediately: hold_frames=1) ----
density = TrafficDensity(roi=FULL, low_max=0.08, high_min=0.18, smooth=1.0, hold_frames=1)

r = density.update([], shape)
print("empty road     :", r)
assert r["status"] == "Low" and r["in_roi"] == 0

r = density.update([box(0, 0, 30, 40)], shape)          # 1200 / 10000 = 12%
print("one 30x40 box  :", r)
assert abs(r["occupancy"] - 0.12) < 0.001 and r["status"] == "Medium"

r = density.update([box(0, 0, 50, 50)], shape)          # 25%
print("one 50x50 box  :", r)
assert abs(r["occupancy"] - 0.25) < 0.001 and r["status"] == "High"

r = density.update([box(0, 0, 50, 50), box(0, 0, 50, 50)], shape)   # same box twice
print("same box twice :", r)
assert abs(r["occupancy"] - 0.25) < 0.001 and r["in_roi"] == 2

# ---- Part 2: the hold time (a new status must last 3 frames) ----
hold = TrafficDensity(roi=FULL, low_max=0.08, high_min=0.18, smooth=1.0, hold_frames=3)
big = [box(0, 0, 50, 50)]

hold.update([], shape)                                   # first frame sets "Low"
assert hold.update(big, shape)["status"] == "Low"        # 1st frame of High: not yet
assert hold.update(big, shape)["status"] == "Low"        # 2nd: not yet
assert hold.update(big, shape)["status"] == "High"       # 3rd: switches
print("hold time      : status switched on the 3rd frame, as expected")

assert hold.update([], shape)["status"] == "High"        # one empty frame: ignored
assert hold.update(big, shape)["status"] == "High"       # back to normal
print("short blip     : ignored, status stayed High")

print("All density checks passed.")