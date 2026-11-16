"""Compare the vehicle count of the program with your own hand count.

Usage:
    python tests\\evaluate_counting.py A-B:YOUR_COUNT
    python tests\\evaluate_counting.py cached A-B:YOUR_COUNT

Every argument is  first_second-last_second:your_count  (vehicles that crossed the yellow
line in that part of the video).

The first form plays the whole video once (about 4 minutes) and saves what the program
counted in output\\count_events.json. With the word "cached" that saved result is used, so
you can try other windows or other hand counts within a second.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

cache_file = PROJECT_ROOT / "output" / "count_events.json"
args = sys.argv[1:]
use_cache = "cached" in args
args = [a for a in args if a != "cached"]

windows = []
for arg in args:
    try:
        span, count = arg.split(":")
        first, last = span.split("-")
        windows.append((float(first), float(last), int(count)))
    except ValueError:
        sys.exit(f"Cannot read '{arg}'. Use first-last:count, for example 0-10:14")
if not windows:
    sys.exit(__doc__)


def play_video():
    """Run detection, tracking and counting over the whole video; return the counted events."""
    import cv2

    import config
    from detection.vehicle_counter import VehicleCounter
    from detection.vehicle_detector import VehicleDetector

    cv2.setNumThreads(1)
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
    counter = VehicleCounter(line_ratio=config.LINE_RATIO)

    print("Playing the whole video once, please wait (about 4 minutes) ...", flush=True)
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (width, height))
        counter.update(detector.track(frame), height, index / fps)
        index += 1
        if index % 150 == 0:
            print(f"  {index / fps:5.1f} s of the video done, {counter.stats()['total']} counted so far",
                  flush=True)
    cap.release()
    return [list(e) for e in counter.events]


if use_cache:
    if not cache_file.exists():
        sys.exit("No saved result yet. Run once without the word 'cached'.")
    events = json.loads(cache_file.read_text())
    print("Using the saved result of the last full run.")
else:
    events = play_video()
    cache_file.parent.mkdir(exist_ok=True)
    cache_file.write_text(json.dumps(events))
    print(f"Saved the program's counts in {cache_file}")

print()
print(f"{'Window':<12} {'Hand count':>10} {'Program':>8} {'Difference':>11}")
total_hand = total_program = total_error = 0
details = []
for first, last, hand in windows:
    inside = [e for e in events if first <= e[0] < last]
    program = len(inside)
    total_hand += hand
    total_program += program
    total_error += abs(program - hand)
    print(f"{first:>4.0f}-{last:<5.0f}s  {hand:>10} {program:>8} {program - hand:>+11}")
    if program != hand:
        details.append((first, last, inside))

accuracy = max(0.0, 1 - total_error / total_hand) if total_hand else 0.0
print(f"{'Total':<12} {total_hand:>10} {total_program:>8} {total_program - total_hand:>+11}")
print(f"\nCount accuracy = 1 - sum(|program - hand|) / sum(hand) = {accuracy:.0%}")
if total_hand:
    print(f"The program found {total_program / total_hand:.0%} as many vehicles as you counted.")

for first, last, found in details:
    print(f"\nProgram events in {first:.0f}-{last:.0f} s (time, vehicle id, type, direction):")
    print("  " + ", ".join(f"{t}s #{vid} {label} {direction}" for t, vid, label, direction in found))
print("\nNote: a vehicle counted a moment before or after a window border can move one count")
print("to the neighbouring window, so differences of 1 per window are within hand-count error.")