"""Run the pipeline WITHOUT the web server and print its status for 90 seconds."""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.pipeline import TrafficPipeline

pipeline = TrafficPipeline()
pipeline.start()
start = time.time()
while time.time() - start < 90:
    time.sleep(3)
    s = pipeline.get_status()
    print(f"{time.time() - start:4.0f} s | {s['status']:<8} | {s['message']:<38} | "
          f"video {s['video']['time']:>5} s | {s['video']['fps']} FPS | "
          f"vehicles {s['vehicles']['total']}", flush=True)
    if s["status"] == "error":
        break
pipeline.stop()
print("Done.")