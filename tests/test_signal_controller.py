"""Quick check of SignalController with made-up statuses and times."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from detection.signal_controller import SignalController

# ---- Green time per status ----
ctrl = SignalController(min_green=15, max_green=45, yellow=3, red=20)
print("green times:", {s: ctrl.green_for(s) for s in ("Low", "Medium", "High")})
assert [ctrl.green_for(s) for s in ("Low", "Medium", "High")] == [15, 30, 45]

small = SignalController(min_green=10, max_green=20)
assert [small.green_for(s) for s in ("Low", "Medium", "High")] == [10, 15, 20]

# ---- One full cycle ----
info = ctrl.update("Low", 0)
print("t=0    :", info)
assert info["phase"] == "GREEN" and info["phase_length"] == 15

info = ctrl.update("High", 10)            # traffic got worse, but this green keeps its length
print("t=10   :", info)
assert info["phase"] == "GREEN" and abs(info["time_left"] - 5) < 0.01
assert info["next_green"] == 45           # ...the NEXT green will be long

info = ctrl.update("High", 15.5)
print("t=15.5 :", info)
assert info["phase"] == "YELLOW" and abs(info["time_left"] - 2.5) < 0.01

info = ctrl.update("High", 18.5)
print("t=18.5 :", info)
assert info["phase"] == "RED" and abs(info["time_left"] - 19.5) < 0.01

info = ctrl.update("High", 38.2)          # red is over: new green, sized for High traffic
print("t=38.2 :", info)
assert info["phase"] == "GREEN" and info["phase_length"] == 45
assert abs(info["time_left"] - 44.8) < 0.01

# ---- A big time jump must not break anything ----
info = ctrl.update("Low", 500)
print("t=500  :", info)
assert info["phase"] in ("GREEN", "YELLOW", "RED")
assert 0 <= info["time_left"] <= info["phase_length"]

# ---- Emergency priority ----
print("--- emergency priority ---")
p = SignalController(min_green=15, max_green=45, yellow=3, red=20,
                     min_red=5, hold_left=8, max_priority_green=60)
p.update("Low", 0)                         # green 0-15
p.update("Low", 15.5)                      # yellow 15-18
info = p.update("Low", 18.5)               # red starts at 18
assert info["phase"] == "RED"

info = p.update("Low", 20, emergency=True)        # only 2 s of red so far
print("red 2 s   :", info["phase"], "|", info["priority_note"])
assert info["phase"] == "RED" and "red will end soon" in info["priority_note"]

info = p.update("Low", 23.5, emergency=True)      # 5.5 s of red: red ends early
print("red 5.5 s :", info["phase"], "|", info["priority_note"])
assert info["phase"] == "GREEN" and abs(info["time_left"] - 15) < 0.01
assert "red ended early" in info["priority_note"]

info = p.update("Low", 35, emergency=True)        # green is nearly over: it gets extended
print("green near end:", info["phase"], f"{info['time_left']:.1f}s left |", info["priority_note"])
assert info["phase"] == "GREEN" and abs(info["time_left"] - 8) < 0.01

info = p.update("Low", 50)                        # emergency is over: normal cycle goes on
print("after     :", info["phase"], "|", repr(info["priority_note"]))
assert info["phase"] == "RED" and info["priority_note"] == ""

# green already at its maximum: cannot be extended
cap = SignalController(min_green=15, max_green=45, min_red=5, hold_left=8, max_priority_green=20)
cap.update("High", 0)                              # green for 45 s
info = cap.update("High", 40, emergency=True)      # 5 s left, but 40 s is beyond the 20 s limit
print("at maximum:", info["priority_note"])
assert "maximum" in info["priority_note"] and abs(info["time_left"] - 5) < 0.01

# yellow always finishes first
y = SignalController(min_green=15, max_green=45, yellow=3, red=20)
y.update("Low", 0)
info = y.update("Low", 16, emergency=True)
print("yellow    :", info["phase"], "|", info["priority_note"])
assert info["phase"] == "YELLOW" and "yellow" in info["priority_note"]

print("Priority events:", p.events)
print("All signal checks passed.")