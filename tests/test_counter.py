"""Tests for the vehicle counter: crossing, ID bridging and the duplicate filter.

Run:  python tests\\test_counter.py
Uses made-up detections (no video, no model), so it takes a fraction of a second.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from detection.vehicle_counter import VehicleCounter

FRAME_HEIGHT = 1000              # line_ratio 0.4 -> the counting line is at y = 400

checks = 0


def check(condition, text):
    """Count one assert; stop with a message if it fails."""
    global checks
    checks += 1
    if not condition:
        sys.exit(f"FAIL: {text}")


def det(vid, cx, cy, label="motorcycle", w=40, h=60):
    """One tracked detection: id, a box around (cx, cy), and the type."""
    return {"id": vid, "bbox": (cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2), "label": label}


def new_counter():
    return VehicleCounter(line_ratio=0.4)


# 1. a vehicle crossing the line is counted once
c = new_counter()
for t, cy in enumerate([500, 450, 390, 340, 300]):
    c.update([det(1, 500, cy)], FRAME_HEIGHT, t / 30)
check(c.total == 1, "one crossing = one count")
check(c.up == 1 and c.down == 0, "going up is counted as up")
check(c.events[0][1:] == (1, "motorcycle", "up"), "event holds id, type and direction")

# 2. going down is counted as down
c = new_counter()
for t, cy in enumerate([300, 350, 410, 460]):
    c.update([det(1, 500, cy)], FRAME_HEIGHT, t / 30)
check(c.total == 1 and c.down == 1, "going down is counted as down")

# 3. no crossing, no count
c = new_counter()
for t, cy in enumerate([700, 650, 600, 560]):
    c.update([det(1, 500, cy)], FRAME_HEIGHT, t / 30)
check(c.total == 0, "a vehicle that never reaches the line is not counted")

c = new_counter()
for t, cy in enumerate([350, 300, 250]):
    c.update([det(1, 500, cy)], FRAME_HEIGHT, t / 30)
check(c.total == 0, "a vehicle first seen past the line is not counted")

# 4. detections without an ID are ignored
c = new_counter()
c.update([det(None, 500, 450)], FRAME_HEIGHT)
c.update([det(None, 500, 350)], FRAME_HEIGHT)
check(c.total == 0, "detections without an ID are ignored")

# 5. ID bridging: the tracker loses a bike just before the line and gives it a new ID after it
c = new_counter()
for cy in [500, 470, 440]:
    c.update([det(1, 500, cy)], FRAME_HEIGHT)
c.update([], FRAME_HEIGHT)
c.update([], FRAME_HEIGHT)
c.update([det(2, 500, 380)], FRAME_HEIGHT)
check(c.total == 1, "a crossing split over two IDs is counted once")
check(c.bridged == 1, "the new ID was linked to the lost track")

# 6. bridging must not link things that do not belong together
c = new_counter()
for cy in [500, 470, 440]:
    c.update([det(1, 500, cy)], FRAME_HEIGHT)
c.update([], FRAME_HEIGHT)
c.update([det(2, 800, 380)], FRAME_HEIGHT)
check(c.bridged == 0, "a new ID far to the side is not linked")
check(c.total == 0, "so no crossing is invented")

c = new_counter()
c.update([det(1, 500, 440)], FRAME_HEIGHT)
for _ in range(25):
    c.update([], FRAME_HEIGHT)
c.update([det(2, 500, 380)], FRAME_HEIGHT)
check(c.bridged == 0, "a track lost for too long is forgotten")

c = new_counter()
for cy in [500, 470, 440]:
    c.update([det(1, 500, cy, "motorcycle")], FRAME_HEIGHT)
c.update([det(2, 500, 380, "bus", 200, 300)], FRAME_HEIGHT)
check(c.bridged == 0, "a different vehicle type is not linked")

# 7. a bike that gets a new ID AFTER it was counted is not counted again
c = new_counter()
c.update([det(1, 500, 450)], FRAME_HEIGHT)
c.update([det(1, 500, 390)], FRAME_HEIGHT)          # counted here
c.update([], FRAME_HEIGHT)
c.update([det(2, 500, 360)], FRAME_HEIGHT)          # the same bike with a new ID
c.update([det(2, 500, 330)], FRAME_HEIGHT)
check(c.total == 1, "a new ID after the count does not count again")

# 8. duplicate filter: two boxes on one vehicle at the same moment
c = new_counter()
c.update([det(1, 500, 450), det(2, 505, 455)], FRAME_HEIGHT)
c.update([det(1, 500, 390), det(2, 505, 395)], FRAME_HEIGHT)
check(c.total == 1, "two boxes on one bike at the same moment give one count")

c = new_counter()
c.update([det(1, 400, 450), det(2, 600, 455)], FRAME_HEIGHT)
c.update([det(1, 400, 390), det(2, 600, 395)], FRAME_HEIGHT)
check(c.total == 2, "two bikes side by side are two counts")


# 9. the duplicate window depends on the type: long for trucks (cab + body), short for bikes
def one_behind_other(label):
    """id1 crosses early and leaves; id2 (a little to the side) crosses 18 frames later."""
    counter = new_counter()
    for f in range(1, 21):
        dets = []
        if f == 1:
            dets.append(det(1, 500, 450, label))
        if f == 2:
            dets.append(det(1, 500, 390, label))
        cy2 = 470 if f <= 18 else (410 if f == 19 else 395)
        dets.append(det(2, 520, cy2, label))
        counter.update(dets, FRAME_HEIGHT)
    return counter.total


check(one_behind_other("truck") == 1, "a truck cab and its body 18 frames apart are one count")
check(one_behind_other("motorcycle") == 2, "two bikes 18 frames apart are two counts")

# 10. counts per type and the stats keys the dashboard reads
c = new_counter()
for t, cy in enumerate([500, 390]):
    c.update([det(1, 300, cy, "car", 100, 80)], FRAME_HEIGHT, t)
for t, cy in enumerate([500, 390]):
    c.update([det(2, 700, cy, "bus", 200, 300)], FRAME_HEIGHT, 2 + t)
check(c.by_type == {"car": 1, "bus": 1}, "counts are kept per type")
check(c.stats()["total"] == 2, "stats() reports the total")
check(set(c.stats()) == {"live", "smoothed", "total", "down", "up", "by_type"},
      "stats() keys are unchanged (the dashboard reads them)")

print(f"Counter tests passed: {checks} asserts")