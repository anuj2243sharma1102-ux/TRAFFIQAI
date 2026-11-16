"""Make a clean video for counting vehicles by hand: only the counting line and a clock.

Usage:
    python tests\\make_count_video.py [window_seconds]
Output: output\\count_video.mp4
"""

import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config

window = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0

cap = cv2.VideoCapture(str(config.VIDEO_SOURCE))
if not cap.isOpened():
    sys.exit("Could not open the video.")
fps = cap.get(cv2.CAP_PROP_FPS) or 25
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out_w = 540
out_h = int(height * out_w / width)

out_path = PROJECT_ROOT / "output" / "count_video.mp4"
writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))

index = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.resize(frame, (out_w, out_h))
    t = index / fps
    line_y = int(out_h * config.LINE_RATIO)
    cv2.line(frame, (0, line_y), (out_w, line_y), (0, 255, 255), 2)
    start = int(t // window) * window
    cv2.rectangle(frame, (0, 0), (out_w, 34), (0, 0, 0), -1)
    cv2.putText(frame, f"{t:5.1f} s    window {start:.0f}-{start + window:.0f} s", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    writer.write(frame)
    index += 1

cap.release()
writer.release()
print("Saved:", out_path)