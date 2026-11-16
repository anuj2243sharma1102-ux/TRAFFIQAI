"""Quick check of ViolationDetector with made-up vehicle tracks (no video needed)."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from detection.violation_detector import ViolationDetector

shape = (400, 300, 3)          # frame height 400, so a stop line at 40% is y = 160


def det(vid, cx, cy):
    return {"id": vid, "label": "car", "confidence": 0.9,
            "bbox": (cx - 10, cy - 10, cx + 10, cy + 10)}


# ---- wrong way and stopped ----
v = ViolationDetector(flow="up", wrong_way_pixels=40, wrong_way_checks=5,
                      stop_seconds=3, stop_pixels=5)
t = 0.0
for i in range(12):
    v.update([det(1, 100, 100 + 10 * i),     # id 1 moves DOWN   -> wrong way
              det(2, 150, 300 - 10 * i),     # id 2 moves UP     -> normal
              det(3, 200, 250)],             # id 3 stands still -> stopped
             shape, t, density_status="Low")
    t += 0.5

found = {(e["id"], e["type"]) for e in v.events}
print("events:", v.events)
assert (1, "wrong_way") in found, "id 1 should be a wrong-way vehicle"
assert (3, "stopped") in found, "id 3 should be a stopped vehicle"
assert not any(e["id"] == 2 for e in v.events), "id 2 drives normally"
assert sum(1 for e in v.events if e["id"] == 1 and e["type"] == "wrong_way") == 1, "report only once"

# ---- standing still in a traffic jam is NOT a violation ----
jam = ViolationDetector(stop_seconds=3, stop_pixels=5)
t = 0.0
for i in range(12):
    jam.update([det(4, 200, 250)], shape, t, density_status="High")
    t += 0.5
print("jam events:", jam.events)
assert jam.events == []

# ---- red light: crossing the line at y=160 going up ----
def run_red(phase):
    r = ViolationDetector(flow="up", line_ratio=0.4)
    for cy in (190, 170, 150, 130):
        r.update([det(5, 100, cy)], shape, 0.0, signal_phase=phase, check_red_light=True)
    return r.events

red_events = run_red("RED")
green_events = run_red("GREEN")
print("on RED  :", red_events)
print("on GREEN:", green_events)
assert len(red_events) == 1 and red_events[0]["type"] == "red_light"
assert green_events == []

# ---- red-light rule is off unless asked for ----
off = ViolationDetector(flow="up", line_ratio=0.4)
for cy in (190, 170, 150, 130):
    off.update([det(6, 100, cy)], shape, 0.0, signal_phase="RED", check_red_light=False)
assert off.events == []

print("All violation checks passed.")