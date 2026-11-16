"""Traffic violation detection from tracked vehicles (rule based).

Rules:
  wrong_way : a vehicle keeps moving AGAINST the normal flow of traffic
  stopped   : a vehicle stays (almost) still for too long on the road
              (ignored while traffic is heavy, because then everybody is stopped)
  red_light : a vehicle crosses the stop line while the signal is RED
              (needs the REAL signal state; with our simulated signal it is a demo only)

Each violation is reported only once per vehicle ID.
"""

from collections import Counter, deque

import cv2

LABELS = {"wrong_way": "WRONG WAY", "stopped": "STOPPED", "red_light": "RED LIGHT"}
COLORS = {"wrong_way": (0, 0, 255), "stopped": (0, 165, 255), "red_light": (255, 0, 255)}


class ViolationDetector:
    def __init__(self, flow="up", line_ratio=0.4, wrong_way_pixels=80, wrong_way_checks=15,
                 stop_seconds=8.0, stop_pixels=15, ignore_left=0.0, show_seconds=3.0):
        self.flow = flow                          # normal direction of traffic: "up" or "down"
        self.line_ratio = line_ratio              # stop line position (fraction of frame height)
        self.wrong_way_pixels = wrong_way_pixels  # how far it must move the wrong way ...
        self.wrong_way_checks = wrong_way_checks  # ... within this many frames
        self.stop_seconds = stop_seconds          # standing still for this long = stopped
        self.stop_pixels = stop_pixels            # movement below this counts as "not moving"
        self.ignore_left = ignore_left            # ignore the left part of the frame (kerb parking)
        self.show_seconds = show_seconds          # how long the red box stays on screen

        self.track = {}        # vehicle id -> its movement history
        self.reported = set()  # (id, rule) pairs already reported
        self.events = []       # every violation found
        self._boxes = {}       # boxes of the current frame, by id

    def _report(self, kind, det, now, new_events):
        key = (det["id"], kind)
        if key in self.reported:
            return
        self.reported.add(key)
        event = {"type": kind, "id": det["id"], "label": det["label"],
                 "time": round(now, 1), "bbox": det["bbox"]}
        self.events.append(event)
        new_events.append(event)

    def update(self, detections, frame_shape, now, density_status="Low",
               signal_phase=None, check_red_light=False):
        """Call once per frame. Returns the NEW violations found in this frame."""
        height, width = frame_shape[:2]
        line_y = int(height * self.line_ratio)
        new_events = []
        self._boxes = {d["id"]: d["bbox"] for d in detections if d["id"] is not None}

        for det in detections:
            vid = det["id"]
            if vid is None:
                continue
            x1, y1, x2, y2 = det["bbox"]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            st = self.track.get(vid)
            if st is None:
                st = {"ys": deque(maxlen=self.wrong_way_checks + 1), "last_y": None,
                      "anchor": (cx, cy), "anchor_time": now}
                self.track[vid] = st
            st["ys"].append(cy)

            # --- wrong way: moved far against the normal flow in a short time ---
            if len(st["ys"]) == st["ys"].maxlen:
                moved = st["ys"][-1] - st["ys"][0]          # + means downwards in the picture
                against = moved if self.flow == "up" else -moved
                if against >= self.wrong_way_pixels:
                    self._report("wrong_way", det, now, new_events)

            # --- stopped: no real movement for stop_seconds (and not a traffic jam) ---
            ax, ay = st["anchor"]
            if abs(cx - ax) + abs(cy - ay) > self.stop_pixels:
                st["anchor"], st["anchor_time"] = (cx, cy), now      # it moved: start counting again
            elif (now - st["anchor_time"] >= self.stop_seconds
                  and density_status != "High" and cx >= self.ignore_left * width):
                self._report("stopped", det, now, new_events)

            # --- red light: crossed the stop line while the signal is RED ---
            if check_red_light and signal_phase == "RED" and st["last_y"] is not None:
                prev = st["last_y"]
                if self.flow == "up":
                    crossed = prev >= line_y > cy
                else:
                    crossed = prev < line_y <= cy
                if crossed:
                    self._report("red_light", det, now, new_events)
            st["last_y"] = cy

        return new_events

    def summary(self):
        return dict(Counter(e["type"] for e in self.events))

    def draw(self, frame, now, show_bar=True):
        """Box + label on recent violators, and (optionally) a counter above the status bar."""
        h, w = frame.shape[:2]
        for ev in self.events:
            if now - ev["time"] > self.show_seconds:
                continue
            bbox = self._boxes.get(ev["id"])
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            color = COLORS[ev["type"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
            cv2.putText(frame, LABELS[ev["type"]], (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if not show_bar:
            return frame

        counts = self.summary()
        text = "Violations: " + (", ".join(f"{LABELS[k]} {v}" for k, v in counts.items())
                                 if counts else "0")
        cv2.rectangle(frame, (0, h - 140), (w, h - 112), (0, 0, 0), -1)
        cv2.putText(frame, text, (8, h - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return frame