"""Step 1 of the density check by eye: save clean pictures, hide the program's answer.

Usage:
    python tests\\density_check_frames.py

Saves 12 pictures (output\\density_check\\01.jpg ...) taken at fixed moments of the video,
with only the region of interest drawn on them (the band between the two cyan lines).
The program's own Low / Medium / High for each moment is saved in
output\\density_check\\program_status.json. Do NOT open it before you have written down
your own answer for every picture.
"""

import json
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from detection.traffic_density import TrafficDensity
from detection.vehicle_detector import VehicleDetector

cv2.setNumThreads(1)

MOMENTS = [1.5, 4.0, 6.5, 9.0, 11.5, 14.0, 16.5, 19.0, 21.5, 24.0, 26.5, 29.0]   # seconds

folder = PROJECT_ROOT / "output" / "density_check"
folder.mkdir(parents=True, exist_ok=True)
for old in folder.glob("*"):
    if old.is_file():
        old.unlink()

cap = cv2.VideoCapture(str(config.VIDEO_SOURCE))
if not cap.isOpened():
    sys.exit("Could not open the video.")
fps = cap.get(cv2.CAP_PROP_FPS) or 25
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
if width > config.MAX_WIDTH:
    height = int(height * config.MAX_WIDTH / width)
    width = config.MAX_WIDTH

detector = VehicleDetector(model_path=config.VEHICLE_MODEL, confidence=config.CONFIDENCE,
                           image_size=config.IMAGE_SIZE)
density = TrafficDensity()
roi_top = int(height * density.roi[0][1])
roi_bottom = int(height * density.roi[2][1])

targets = {round(t * fps): (n, t) for n, t in enumerate(MOMENTS, 1)}
last_index = max(targets)
program = []

print(f"Playing the video up to {MOMENTS[-1]:.1f} s (about 4 minutes) ...", flush=True)
index = 0
while index <= last_index:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.resize(frame, (width, height))
    result = density.update(detector.track(frame), frame.shape)
    if index in targets:
        n, t = targets[index]
        picture = frame.copy()
        cv2.line(picture, (0, roi_top), (width, roi_top), (255, 255, 0), 3)
        cv2.line(picture, (0, roi_bottom), (width, roi_bottom), (255, 255, 0), 3)
        small = cv2.resize(picture, (540, int(height * 540 / width)))
        cv2.imwrite(str(folder / f"{n:02d}.jpg"), small)
        program.append({"n": n, "time": t, "status": result["status"],
                        "occupancy": result["smoothed"]})
        print(f"  picture {n:02d} saved", flush=True)
    index += 1
cap.release()

(folder / "program_status.json").write_text(json.dumps(program, indent=1))
print(f"\nDone. Look at the pictures in {folder}")
print("Write down Low / Medium / High for each one BEFORE opening program_status.json.")