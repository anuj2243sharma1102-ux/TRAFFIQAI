"""Save every N-th frame of a video as a JPG.
Used to build our own training data (positives and negatives).

Usage:
    python tools\\extract_frames.py <video> <output_folder> [every_n] [start_s] [end_s]

Example:
    python tools\\extract_frames.py videos\\amb_fire.mp4 datasets\\amb_fire_train 8 0 14
"""

import sys
from pathlib import Path

import cv2

if len(sys.argv) < 3:
    sys.exit(__doc__)

video = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
every_n = int(sys.argv[3]) if len(sys.argv) > 3 else 10
start_s = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
end_s = float(sys.argv[5]) if len(sys.argv) > 5 else 1e9

if not video.exists():
    sys.exit(f"File not found: {video}")

cap = cv2.VideoCapture(str(video))
fps = cap.get(cv2.CAP_PROP_FPS) or 25
out_dir.mkdir(parents=True, exist_ok=True)

frame_idx = 0
saved = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    t = frame_idx / fps
    if t >= end_s:
        break
    if t >= start_s and frame_idx % every_n == 0:
        scale = 1280 / max(frame.shape[:2])      # keep the longer side at 1280 px
        if scale < 1:
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
        cv2.imwrite(str(out_dir / f"{video.stem}_{frame_idx:05d}.jpg"), frame)
        saved += 1
    frame_idx += 1

cap.release()
print(f"Saved {saved} frames to {out_dir}")