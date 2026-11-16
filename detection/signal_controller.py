"""Adaptive traffic signal for ONE approach (the road the camera watches).

Cycle: GREEN -> YELLOW -> RED -> GREEN ...
The length of each GREEN is decided when it starts, from the traffic status:
    Low -> min_green,  Medium -> halfway,  High -> max_green

Emergency priority: when an emergency vehicle is reported (emergency=True)
    RED    : after min_red seconds the red is ended early and green starts
    GREEN  : the green is held so that at least hold_left seconds remain
             (but never longer than max_priority_green in total)
    YELLOW : always finishes first (safety), then the RED rule applies

Time is passed in by the caller (video time in seconds), so the controller
works the same for videos, live cameras and unit tests.
"""

import cv2

# How much of the allowed green range each traffic status gets (0 = min, 1 = max)
LEVELS = {"Low": 0.0, "Medium": 0.5, "High": 1.0}

NEXT_PHASE = {"GREEN": "YELLOW", "YELLOW": "RED", "RED": "GREEN"}

# Lamp colours in BGR
PHASE_COLORS = {"GREEN": (0, 200, 0), "YELLOW": (0, 215, 255), "RED": (0, 0, 230)}


class SignalController:
    def __init__(self, min_green=15, max_green=45, yellow=3, red=20,
                 min_red=5, hold_left=8, max_priority_green=60):
        self.min_green = min_green
        self.max_green = max_green
        self.yellow = yellow
        self.red = red
        self.min_red = min_red                          # red must last this long before priority can end it
        self.hold_left = hold_left                      # keep at least this much green left during an emergency
        self.max_priority_green = max_priority_green    # never hold a green longer than this

        self.phase = None           # "GREEN", "YELLOW" or "RED"
        self.phase_start = None     # time when this phase began
        self.phase_length = None    # how long this phase lasts
        self.log = []               # (start time, phase, length, status) for reports
        self.events = []            # (time, note) every time the priority note changes
        self._last_note = ""

    def green_for(self, status):
        """Green time (seconds) the given traffic status deserves."""
        level = LEVELS.get(status, 0.5)
        return round(self.min_green + (self.max_green - self.min_green) * level)

    def _length_for(self, phase, status):
        if phase == "GREEN":
            return self.green_for(status)
        if phase == "YELLOW":
            return self.yellow
        return self.red

    def _start_phase(self, phase, start, status):
        self.phase = phase
        self.phase_start = start
        self.phase_length = self._length_for(phase, status)
        self.log.append((round(start, 1), phase, self.phase_length, status))

    def update(self, status, now, emergency=False):
        """Call once per frame with the traffic status, time (s) and emergency flag."""
        if self.phase is None:
            self._start_phase("GREEN", now, status)

        # Normal cycle: move on to the next phase(s) if the current one is over
        while now - self.phase_start >= self.phase_length:
            self._start_phase(NEXT_PHASE[self.phase],
                              self.phase_start + self.phase_length, status)

        # Emergency priority
        note = ""
        if emergency:
            elapsed = now - self.phase_start
            if self.phase == "RED":
                if elapsed >= self.min_red:
                    self._start_phase("GREEN", now, status)
                    note = "PRIORITY: red ended early, green for emergency vehicle"
                else:
                    note = "PRIORITY: red will end soon"
            elif self.phase == "GREEN":
                note = "PRIORITY: green active for emergency vehicle"
                left = self.phase_length - elapsed
                if left < self.hold_left:
                    if elapsed + self.hold_left <= self.max_priority_green:
                        self.phase_length = elapsed + self.hold_left    # extend the green
                    else:
                        note = "PRIORITY: green already at its maximum"
            else:
                note = "PRIORITY: waiting for yellow to finish"

        if note != self._last_note:
            if note:
                self.events.append((round(now, 1), note))
            self._last_note = note

        return {
            "phase": self.phase,
            "time_left": self.phase_start + self.phase_length - now,
            "phase_length": self.phase_length,
            "next_green": self.green_for(status),   # what the next green would be now
            "status": status,
            "priority_note": note,
        }

    def draw(self, frame, info):
        """Draw a small traffic light and the countdown on the right side."""
        h, w = frame.shape[:2]
        x, y0 = w - 46, 125

        cv2.rectangle(frame, (x - 26, y0 - 16), (x + 26, y0 + 118), (30, 30, 30), -1)
        for name, cy in (("RED", y0 + 12), ("YELLOW", y0 + 54), ("GREEN", y0 + 96)):
            color = PHASE_COLORS[name] if info["phase"] == name else (60, 60, 60)
            cv2.circle(frame, (x, cy), 16, color, -1)

        seconds_left = max(0, int(info["time_left"] + 0.999))
        cv2.rectangle(frame, (w - 200, y0 + 126), (w, y0 + 178), (0, 0, 0), -1)
        cv2.putText(frame, f"{info['phase']} {seconds_left}s", (w - 194, y0 + 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, PHASE_COLORS[info["phase"]], 2)
        cv2.putText(frame, f"Next green: {info['next_green']}s", (w - 194, y0 + 172),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if info.get("priority_note"):
            cv2.rectangle(frame, (w - 200, y0 + 182), (w, y0 + 206), (0, 0, 200), -1)
            cv2.putText(frame, "EMERGENCY PRIORITY", (w - 196, y0 + 199),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        return frame