"""Find out where reading / resizing video frames blocks.

Usage:
    python tests\\diagnose_video.py        (normal)
    python tests\\diagnose_video.py 1      (OpenCV limited to 1 thread)
"""

import faulthandler
import sys
import threading
import time
from pathlib import Path

import cv2

faulthandler.dump_traceback_later(100, exit=True)   # if something blocks: show where, then quit

PROJECT_ROOT = Path(__file__).resolve().parent.parent
video = PROJECT_ROOT / "videos" / "cctv.mp4"

if len(sys.argv) > 1 and sys.argv[1] == "1":
    cv2.setNumThreads(1)
    print("OpenCV limited to 1 thread", flush=True)
print("OpenCV version:", cv2.__version__, "| threads:", cv2.getNumThreads(), flush=True)


def report(text, start):
    print(f"  {text:<44} {time.time() - start:6.2f} s", flush=True)


def run_in_thread(name, function, timeout=40):
    """Run function in a worker thread and say clearly if it does not finish."""
    thread = threading.Thread(target=function, daemon=True)
    thread.start()
    thread.join(timeout)
    finished = not thread.is_alive()
    print(f"  -> {name}: " + ("finished" if finished else f"STUCK (no end after {timeout} s)"), flush=True)
    return finished


print("1) main thread: open, read, resize", flush=True)
t0 = time.time()
cap = cv2.VideoCapture(str(video))
report("open video", t0)
ok, frame = cap.read()
report(f"read first frame (ok={ok})", t0)
if not ok:
    sys.exit("Could not read a frame from the video.")
cv2.resize(frame, (720, 1280))
report("resize", t0)


print("2) worker thread: read and resize (no AI models loaded yet)", flush=True)


def worker_without_models():
    t = time.time()
    ok2, f2 = cap.read()
    report("read frame", t)
    cv2.resize(f2, (720, 1280))
    report("resize", t)


run_in_thread("worker without models", worker_without_models)


print("3) load PyTorch and YOLO, run one detection (main thread)", flush=True)
t0 = time.time()
from ultralytics import YOLO

report("import ultralytics", t0)
model = YOLO(str(PROJECT_ROOT / "models" / "yolov8n.pt"))
report("load yolov8n", t0)
model(frame, imgsz=960, verbose=False)
report("one detection", t0)


print("4) worker thread again, now AFTER PyTorch was loaded (like the real pipeline)", flush=True)


def worker_after_torch():
    t = time.time()
    ok3, f3 = cap.read()
    report("read frame", t)
    cv2.resize(f3, (720, 1280))
    report("resize", t)
    model.track(f3, persist=True, imgsz=960, verbose=False)
    report("track", t)


run_in_thread("worker after PyTorch", worker_after_torch, timeout=90)
print("Done.", flush=True)