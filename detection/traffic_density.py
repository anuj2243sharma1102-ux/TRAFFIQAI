"""Traffic density: how full is the road?

Input : detections of one frame (from VehicleDetector)
Output: occupancy (share of the road area covered by vehicles),
        number of vehicles in the region of interest, and a status
        of Low / Medium / High.
"""

import cv2
import numpy as np

# Region of interest as fractions (0..1) of the frame: (x, y) points.
# Default = a band across the road, skipping the top (tiny far vehicles)
# and the bottom (vehicles right under the camera, often not detected).
DEFAULT_ROI = [(0.0, 0.15), (1.0, 0.15), (1.0, 0.55), (0.0, 0.55)]

# Status colours in BGR
STATUS_COLORS = {
    "Low": (0, 200, 0),
    "Medium": (0, 165, 255),
    "High": (0, 0, 255),
}


class TrafficDensity:
    def __init__(self, roi=None, low_max=0.13, high_min=0.20, smooth=0.1, hold_frames=30):
        self.roi = roi or DEFAULT_ROI
        self.low_max = low_max          # below this occupancy  -> Low
        self.high_min = high_min        # at or above this      -> High
        self.alpha = smooth             # 0.1 = slow and steady, 1.0 = no smoothing
        self.hold_frames = hold_frames  # frames a new status must last before it switches
        self.smoothed = None
        self.status = "Low"
        self._pending = None            # status waiting to take over
        self._pending_count = 0

    def _roi_points(self, width, height):
        return np.array(
            [(int(x * width), int(y * height)) for x, y in self.roi], dtype=np.int32
        )

    def _raw_status(self, value):
        if value < self.low_max:
            return "Low"
        if value >= self.high_min:
            return "High"
        return "Medium"

    def update(self, detections, frame_shape):
        height, width = frame_shape[:2]

        # 1 = pixel inside the region of interest
        roi_mask = np.zeros((height, width), np.uint8)
        cv2.fillPoly(roi_mask, [self._roi_points(width, height)], 1)
        roi_area = int(roi_mask.sum())

        # 1 = pixel covered by at least one vehicle box (overlaps count once)
        vehicle_mask = np.zeros((height, width), np.uint8)
        in_roi = 0
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cv2.rectangle(vehicle_mask, (x1, y1), (x2 - 1, y2 - 1), 1, -1)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if 0 <= cx < width and 0 <= cy < height and roi_mask[cy, cx]:
                in_roi += 1

        covered = int((vehicle_mask & roi_mask).sum())
        occupancy = covered / roi_area if roi_area else 0.0

        # Smooth the value so it does not jump around
        first_frame = self.smoothed is None
        if first_frame:
            self.smoothed = occupancy
        else:
            self.smoothed = self.alpha * occupancy + (1 - self.alpha) * self.smoothed

        # Change the status only when the new one lasts for hold_frames frames
        raw_status = self._raw_status(self.smoothed)
        if first_frame:
            self.status = raw_status
        elif raw_status == self.status:
            self._pending, self._pending_count = None, 0
        else:
            if raw_status == self._pending:
                self._pending_count += 1
            else:
                self._pending, self._pending_count = raw_status, 1
            if self._pending_count >= self.hold_frames:
                self.status = raw_status
                self._pending, self._pending_count = None, 0

        return {
            "occupancy": occupancy,
            "smoothed": self.smoothed,
            "in_roi": in_roi,
            "status": self.status,
        }

    def draw(self, frame, result):
        """Draw the region of interest and a status bar at the bottom."""
        height, width = frame.shape[:2]
        cv2.polylines(frame, [self._roi_points(width, height)], True, (255, 255, 0), 2)

        color = STATUS_COLORS[result["status"]]
        cv2.rectangle(frame, (0, height - 70), (width, height), (0, 0, 0), -1)
        cv2.putText(frame, f"Traffic: {result['status']}", (8, height - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame,
                    f"Occupancy {result['smoothed'] * 100:.0f}%   In ROI: {result['in_roi']}",
                    (8, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        return frame