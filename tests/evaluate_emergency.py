"""Check the emergency vehicle detector on test cases and print PASS / FAIL.

Usage:
    python tests\\evaluate_emergency.py [model_file]

A case plays a part of a video.
  "detect": a priority vehicle (Ambulance / Fire Engine) must be confirmed at least once.
  "none"  : nothing may be confirmed (every alert would be a false alarm).
"""

import sys
from collections import Counter
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from detection.emergency_detector import EmergencyDetector

model_file = sys.argv[1] if len(sys.argv) > 1 else "emergency2.pt"

CASES = [
    # (name, video, from second, to second, expectation, note)
    ("Ambulance + fire engine convoy", "amb_fire.mp4", 14, 9999, "detect",
     "second part of the clip (not used for training)"),
    ("Normal traffic with red buses and vans", "cctv.mp4", 15, 9999, "none",
     "part of the video not used for training"),
    ("Normal traffic, first part", "cctv.mp4", 0, 15, "none",
     "used as 'no emergency' examples in training"),
]


def play(video_name, from_s, to_s):
    path = PROJECT_ROOT / "videos" / video_name
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    detector = EmergencyDetector(model_path=PROJECT_ROOT / "models" / model_file)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    index = int(from_s * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    frames = flagged = 0
    labels = Counter()
    best = 0.0
    while True:
        ok, frame = cap.read()
        if not ok or index / fps >= to_s:
            break
        priority = [d for d in detector.update(frame) if d["priority"]]
        if priority:
            flagged += 1
            for d in priority:
                labels[d["label"]] += 1
                best = max(best, d["score"])
        frames += 1
        index += 1
    cap.release()
    return frames, flagged, labels, best


print(f"Emergency detector test cases (model: {model_file})\n")
rows = []
for name, video, start, end, expect, note in CASES:
    print(f"running: {name} ...", flush=True)
    result = play(video, start, end)
    if result is None:
        rows.append((name, expect, "video missing", "-", "-", "FAIL"))
        continue
    frames, flagged, labels, best = result
    passed = flagged > 0 if expect == "detect" else flagged == 0
    found = ", ".join(f"{k} ({v})" for k, v in labels.items()) or "nothing"
    rows.append((name, expect, f"{flagged} of {frames} frames", found, f"{best:.2f}",
                 "PASS" if passed else "FAIL"))

print()
print(f"{'Case':<42} {'Expected':<8} {'Frames flagged':<22} {'Found':<36} {'Best':<5} Result")
for r in rows:
    print(f"{r[0]:<42} {r[1]:<8} {r[2]:<22} {r[3]:<36} {r[4]:<5} {r[5]}")
print(f"\n{sum(r[5] == 'PASS' for r in rows)} of {len(rows)} cases passed.")
print("Note: these videos show the same scenes as the training frames, so this is a")
print("check of the trained scenes, not a general accuracy figure.")