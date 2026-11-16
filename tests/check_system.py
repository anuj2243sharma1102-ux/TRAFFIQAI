"""Quick check before a demo: are all packages, models and files in place?

Usage:
    python tests\\check_system.py
"""

import importlib
import socket
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config

results = []


def check(name, ok, hint=""):
    print(f"[{'OK  ' if ok else 'FAIL'}] {name}" + ("" if ok else f"   -> {hint}"))
    results.append(ok)


print("Packages")
for package in ["cv2", "numpy", "torch", "torchvision", "ultralytics", "flask", "lap"]:
    try:
        importlib.import_module(package)
        ok = True
    except Exception:
        ok = False
    check(f"package {package}", ok, "run: pip install -r requirements.txt")

print("\nFiles")
files = [
    ("video", config.VIDEO_SOURCE),
    ("vehicle detector model", config.VEHICLE_MODEL),
    ("vehicle type model", config.VEHICLE_TYPE_MODEL),
    ("emergency model", config.EMERGENCY_MODEL),
    ("dashboard page", PROJECT_ROOT / "frontend" / "index.html"),
    ("dashboard style", PROJECT_ROOT / "frontend" / "css" / "style.css"),
    ("dashboard script", PROJECT_ROOT / "frontend" / "js" / "dashboard.js"),
]
for label, path in files:
    if isinstance(path, int):                        # a webcam number, not a file
        check(f"{label} (camera {path})", True)
        continue
    check(f"{label}: {Path(path).name}", Path(path).exists() and Path(path).stat().st_size > 0,
          f"missing or empty: {path}")

print("\nContents")
try:
    import cv2
    cap = cv2.VideoCapture(config.VIDEO_SOURCE if isinstance(config.VIDEO_SOURCE, int)
                           else str(config.VIDEO_SOURCE))
    ok, frame = cap.read()
    cap.release()
    check(f"video can be read ({frame.shape[1]}x{frame.shape[0]})" if ok else "video can be read",
          ok, "the video file is damaged or has an unsupported codec")
except Exception as exc:
    check("video can be read", False, str(exc))

try:
    import torch
    saved = torch.load(config.VEHICLE_TYPE_MODEL, weights_only=True)
    check(f"vehicle type model classes: {saved['classes']}", True)
except Exception as exc:
    check("vehicle type model can be loaded", False, str(exc))

try:
    from ultralytics import YOLO
    names = list(YOLO(str(config.EMERGENCY_MODEL)).names.values())
    check(f"emergency model classes: {names}", True)
except Exception as exc:
    check("emergency model can be loaded", False, str(exc))

print("\nDatabase and server")
try:
    from backend.services.database import Database
    db = Database(config.DB_PATH)
    db.close()
    check(f"database can be opened ({config.DB_PATH.name})", True)
except Exception as exc:
    check("database can be opened", False, str(exc))

sock = socket.socket()
try:
    sock.bind((config.HOST, config.PORT))
    check(f"port {config.PORT} is free", True)
except OSError:
    check(f"port {config.PORT} is free", False, "another 'python app.py' is probably still running")
finally:
    sock.close()

print()
if all(results):
    print(f"All {len(results)} checks passed. Start the system with:  python app.py")
else:
    print(f"{results.count(False)} of {len(results)} checks FAILED, see the lines marked FAIL above.")