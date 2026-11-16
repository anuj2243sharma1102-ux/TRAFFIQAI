"""Compare speed and vehicle count when frames are skipped (FRAME_STRIDE).

Usage:
    python tests\\benchmark_stride.py [list of strides]

Example:
    python tests\\benchmark_stride.py 1,2,3,4,6

For every stride the whole video is played once, with all AI modules, without the web
server and without the database. Close "python app.py" first (and other heavy programs),
otherwise the timings are not fair. The whole test takes several minutes.
"""

import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from backend.services.pipeline import TrafficPipeline

strides = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else [1, 2, 3, 4, 6]

cap = cv2.VideoCapture(str(config.VIDEO_SOURCE))
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS) or 25
cap.release()
video_seconds = frames / fps
print(f"Video: {frames} frames = {video_seconds:.1f} s at {fps:.0f} fps")
print("Playing the whole video once for each stride, please wait ...", flush=True)

config.LOOP_VIDEO = False          # play the video only once
results = []

for stride in strides:
    config.FRAME_STRIDE = stride
    pipeline = TrafficPipeline()   # no database
    pipeline.start()
    start = time.time()
    while True:
        time.sleep(2)
        status = pipeline.get_status()
        if status["status"] in ("finished", "error"):
            break
    elapsed = time.time() - start
    if status["status"] == "error":
        sys.exit(f"Error: {status['message']}")
    results.append({
        "stride": stride,
        "elapsed": elapsed,
        "total": status["vehicles"]["total"],
        "by_type": status["vehicles"]["by_type"],
        "violations": status["violations"]["total"],
    })
    print(f"  stride {stride}: finished in {elapsed:.0f} s", flush=True)

baseline = results[0]["total"]
print()
print("stride | frames used | time (s) | speed vs real time | vehicles counted | difference | violations")
for r in results:
    used = -(-frames // r["stride"])
    print(f"{r['stride']:>6} | {used:>11} | {r['elapsed']:>8.0f} | {video_seconds / r['elapsed']:>17.2f}x "
          f"| {r['total']:>16} | {r['total'] - baseline:>+10} | {r['violations']:>10}")
print("(difference = compared with the first stride in the list)")
print("Vehicle types counted:")
for r in results:
    print(f"  stride {r['stride']}: {r['by_type']}")