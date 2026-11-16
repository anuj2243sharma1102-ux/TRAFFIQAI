"""Vehicle counting with a virtual line.

Feed it the tracked detections of every frame (call update()).
A vehicle is counted ONCE, when its centre point crosses the line.

ID bridging: the tracker often loses a motorcycle for a few frames and gives it a new ID.
When a NEW id appears close to a track that was lost a moment ago, it is treated as the same
vehicle, so the crossing is still seen and nothing is counted twice.
"""

from collections import deque

import cv2


class VehicleCounter:
    def __init__(self, line_ratio=0.6, smooth_frames=15, max_gap=20, gate_x=60, speed_px=25,
                dedupe_frames=25, dedupe_dx=40, dedupe_dy=60, dedupe_frames_bike=8):
        self.line_ratio = line_ratio          # 0.6 = line at 60% of frame height
        self.max_gap = max_gap                # a lost track can be re-linked for this many frames
        self.gate_x = gate_x                  # max sideways distance (pixels) for re-linking
        self.speed_px = speed_px              # max vertical distance per lost frame (pixels)
        self.frame_no = 0
        self.alias = {}                       # tracker id -> id used for counting
        self.seen = {}                        # counting id -> (cx, cy, frame_no, label)
        self.bridged = 0                      # how many new ids were linked to an old track
        self.last_y = {}                      # counting id -> centre y in last frame
        self.counted_ids = set()              # counting ids already counted
        self.total = 0
        self.by_type = {}                     # e.g. {"car": 5, "bus": 1}
        self.down = 0                         # crossed from top to bottom
        self.up = 0                           # crossed from bottom to top
        self.events = []                      # (time in seconds, id, type, direction)
        self.history = deque(maxlen=smooth_frames)
        self.live = 0                         # vehicles visible right now
        self.smoothed = 0                     # average of the last few frames
        self.dedupe_frames = dedupe_frames    # a count this close in time ...
        self.dedupe_dx = dedupe_dx            # ... and this close sideways (pixels) ...
        self.dedupe_dy = dedupe_dy            # ... and this close vertically ...
        self.dedupe_frames_bike = dedupe_frames_bike   # bikes: only a double detection at the same mome
        self.recent = []                      # ... with the same type is the same vehicle
    
    def _is_duplicate(self, cx, cy, label):
        """True if a vehicle of the same type was counted at almost the same place a moment ago."""
        limit = self.dedupe_frames_bike if label == "motorcycle" else self.dedupe_frames
        for ox, oy, frame, old_label in self.recent:
            if (self.frame_no - frame <= limit and old_label == label
                    and abs(cx - ox) <= self.dedupe_dx and abs(cy - oy) <= self.dedupe_dy):
                return True
        return False

    def _find_old(self, cx, cy, label, present):
        """Look for a track lost a moment ago, close to this new detection."""
        best, best_score = None, None
        for cid, (ox, oy, frame, old_label) in self.seen.items():
            gap = self.frame_no - frame
            if cid in present or gap < 1 or gap > self.max_gap or old_label != label:
                continue
            dx = abs(cx - ox)
            dy = abs(cy - oy)
            if dx > self.gate_x or dy > 30 + self.speed_px * gap:
                continue
            score = dx + dy
            if best is None or score < best_score:
                best, best_score = cid, score
        return best

    def update(self, detections, frame_height, time_s=0.0):
        line_y = int(frame_height * self.line_ratio)
        self.frame_no += 1

        # Live count and a smoothed version that doesn't flicker
        self.live = len(detections)
        self.history.append(self.live)
        self.smoothed = round(sum(self.history) / len(self.history))

        items = []
        present = set()                       # counting ids visible in this frame
        for det in detections:
            vid = det["id"]
            if vid is None:
                continue                      # no ID yet, can't track it
            x1, y1, x2, y2 = det["bbox"]
            items.append((vid, det, (x1 + x2) // 2, (y1 + y2) // 2))
            if vid in self.alias:
                present.add(self.alias[vid])

        for vid, det, cx, cy in items:
            cid = self.alias.get(vid)
            if cid is None:                   # a tracker id we have not seen before
                cid = self._find_old(cx, cy, det["label"], present)
                if cid is None:
                    cid = vid
                else:
                    self.bridged += 1
                self.alias[vid] = cid
                present.add(cid)

            prev = self.last_y.get(cid)
            self.last_y[cid] = cy
            self.seen[cid] = (cx, cy, self.frame_no, det["label"])

            if prev is None or cid in self.counted_ids:
                continue

            crossed_down = prev < line_y <= cy
            crossed_up = prev >= line_y > cy
            if crossed_down or crossed_up:
                self.counted_ids.add(cid)
                if self._is_duplicate(cx, cy, det["label"]):
                    
                    continue                  # same vehicle already counted a moment ago                  # same vehicle already counted a moment ago
                self.recent.append((cx, cy, self.frame_no, det["label"]))
                self.total += 1
                self.by_type[det["label"]] = self.by_type.get(det["label"], 0) + 1
                direction = "down" if crossed_down else "up"
                if crossed_down:
                    self.down += 1
                else:
                    self.up += 1
                self.events.append((round(time_s, 1), vid, det["label"], direction))

        # forget tracks that were lost for too long
        limit = self.frame_no - self.max_gap
        self.seen = {k: v for k, v in self.seen.items() if v[2] >= limit}

        self.recent = [r for r in self.recent if self.frame_no - r[2] <= self.dedupe_frames]
        return self.stats()

    def stats(self):
        return {
            "live": self.live,
            "smoothed": self.smoothed,
            "total": self.total,
            "down": self.down,
            "up": self.up,
            "by_type": dict(self.by_type),
        }

    def draw(self, frame):
        """Draw the counting line and a small stats panel."""
        h, w = frame.shape[:2]
        line_y = int(h * self.line_ratio)
        cv2.line(frame, (0, line_y), (w, line_y), (0, 255, 255), 2)

        lines = [
            f"Live: {self.live} (avg {self.smoothed})",
            f"Crossed: {self.total}  down {self.down} / up {self.up}",
        ]
        if self.by_type:
            lines.append(", ".join(f"{k} {v}" for k, v in self.by_type.items()))

        cv2.rectangle(frame, (0, 0), (w, 30 * len(lines) + 10), (0, 0, 0), -1)
        for i, text in enumerate(lines):
            cv2.putText(frame, text, (8, 28 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return frame