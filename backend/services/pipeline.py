"""The pipeline: runs all AI modules on the video in a background thread.

The web routes only READ the latest results from here (status as a dictionary and
the latest picture as a JPEG). Nothing in this file knows about Flask.
If a database is given, a row of statistics is stored every few seconds and every
alert is stored as well.
"""

import copy
import threading
import time
import traceback
from collections import deque
from pathlib import Path

import cv2
import numpy as np

import config
from detection.emergency_detector import EmergencyDetector
from detection.signal_controller import SignalController
from detection.traffic_density import TrafficDensity
from detection.vehicle_counter import VehicleCounter
from detection.vehicle_detector import VehicleDetector
from detection.vehicle_type_classifier import VehicleTypeClassifier
from detection.violation_detector import ViolationDetector
from backend.services.challan_demo import ChallanDemo, SIMULATED_TYPES, TITLES as CHALLAN_TITLES

# OpenCV's own thread pool can clash with PyTorch's threads on Windows (the pipeline once
# froze inside cv2.resize); one thread is plenty for resizing a video frame.
cv2.setNumThreads(1)

DEBUG_FRAMES = getattr(config, "DEBUG_FIRST_FRAMES", 0)   # print progress for the first frames

RULE_NAMES = {"wrong_way": "Wrong-way vehicle", "stopped": "Stopped vehicle",
              "red_light": "Red-light violation"}


def _log(text):
    print(f"[pipeline] {text}", flush=True)


class TrafficPipeline:
    def __init__(self, database=None):
        self._db = database
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._state = self._empty_state()
        self._events = deque(maxlen=100)
        self._jpeg = None
        self._frame_id = 0
        self._round = 0                       # how many times the video has been played so far
        self._simulated_emergency = False
                # demo e-challans (simulation): made when a violation is found or the dashboard button is pressed
        self._sim_requests = deque(maxlen=10)
        self._challans = (ChallanDemo(database)
                          if database and getattr(config, "ENABLE_CHALLAN_DEMO", False) else None)

    # ------------------------------------------------------------ used by the web routes
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="pipeline")
        self._thread.start()

    def stop(self):
        self._stop.set()

    def get_status(self):
        with self._lock:
            return copy.deepcopy(self._state)

    def get_events(self, limit=20):
        with self._lock:
            return list(self._events)[-limit:][::-1]      # newest first

    def get_frame(self):
        with self._lock:
            return self._frame_id, self._jpeg

    def set_simulated_emergency(self, active):
        self._simulated_emergency = bool(active)
        with self._lock:                      # show the change at once, not at the next frame
            emergency = self._state["emergency"]
            real = any(d["priority"] for d in emergency["detected"])
            emergency["simulated"] = self._simulated_emergency
            emergency["active"] = self._simulated_emergency or real
            
    def request_simulated_violation(self, kind):
        """Called by the web route: the next frame gets a simulated violation of this type."""
        if self._challans is None or kind not in SIMULATED_TYPES:
            return False
        self._sim_requests.append(kind)
        return True

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _empty_state():
        return {
            "status": "starting",
            "message": "Starting ...",
            "video": {"name": "", "time": 0.0, "fps": 0.0, "loops": 0},
            "vehicles": {"live": 0, "smoothed": 0, "total": 0, "down": 0, "up": 0, "by_type": {}},
            "density": {"occupancy": 0.0, "smoothed": 0.0, "in_roi": 0, "status": "Low"},
            "signal": {"phase": "-", "time_left": 0.0, "phase_length": 0, "next_green": 0,
                       "priority_note": "", "time_scale": config.SIGNAL_TIME_SCALE},
            "emergency": {"active": False, "simulated": False, "detected": []},
            "violations": {"total": 0, "by_type": {}},
        }

    def _set_status(self, status, message=""):
        with self._lock:
            self._state["status"] = status
            self._state["message"] = message

    def _add_event(self, video_time, kind, text):
        event = {"video_time": round(video_time, 1), "type": kind, "text": text,
                 "wall_time": time.strftime("%H:%M:%S")}
        with self._lock:
            self._events.append(event)
        if self._db:
            self._db.add_event(event, self._round)

    @staticmethod
    def _draw_guides(image, density):
        """Thin guide lines: the counting line (yellow) and the monitored zone (cyan)."""
        h, w = image.shape[:2]
        line_y = int(h * config.LINE_RATIO)
        cv2.line(image, (0, line_y), (w, line_y), (0, 255, 255), 2)
        zone = np.array([(int(x * w), int(y * h)) for x, y in density.roi], dtype=np.int32)
        cv2.polylines(image, [zone], True, (255, 255, 0), 1)

    @staticmethod
    def _reset_models(detector, classifier, emergency):
        """Forget everything the models remember, so a new round of the video starts clean."""
        predictor = getattr(detector.model, "predictor", None)
        for tracker in getattr(predictor, "trackers", None) or []:
            try:
                tracker.reset()
            except Exception:
                pass
        if classifier:
            classifier.frame_idx = 0
            classifier.last_checked.clear()
            classifier.history.clear()
        if emergency:
            emergency.frame_idx = 0
            emergency.latest = []
            for hits in emergency.hits.values():
                hits.clear()

    # ------------------------------------------------------------ the background thread
    def _run(self):
        try:
            t = time.time()
            self._set_status("loading", "Loading the vehicle detector ...")
            detector = VehicleDetector(model_path=config.VEHICLE_MODEL,
                                       confidence=config.CONFIDENCE,
                                       image_size=config.IMAGE_SIZE)
            _log(f"vehicle detector loaded ({time.time() - t:.1f} s)")

            self._set_status("loading", "Loading the vehicle type classifier ...")
            classifier = (VehicleTypeClassifier(model_path=config.VEHICLE_TYPE_MODEL)
                          if config.ENABLE_TYPES else None)
            _log(f"type classifier loaded ({time.time() - t:.1f} s)")

            self._set_status("loading", "Loading the emergency detector ...")
            emergency = (EmergencyDetector(model_path=config.EMERGENCY_MODEL,
                                           every_n=config.EMERGENCY_EVERY_N)
                         if config.ENABLE_EMERGENCY else None)
            _log(f"emergency detector loaded ({time.time() - t:.1f} s)")
            self._set_status("loading", "Models loaded, opening the video ...")
        except Exception as exc:                      # for example a missing model file
            traceback.print_exc()
            self._set_status("error", f"Could not load a model: {exc}")
            return

        loops = 0
        while not self._stop.is_set():
            try:
                if not self._play_once(detector, classifier, emergency, loops):
                    return
            except Exception as exc:
                traceback.print_exc()
                self._set_status("error", f"Pipeline stopped: {exc}")
                return
            loops += 1
            if not config.LOOP_VIDEO:
                self._set_status("finished", "Video finished")
                return

    def _play_once(self, detector, classifier, emergency, loops):
        source = config.VIDEO_SOURCE
        cap = cv2.VideoCapture(source if isinstance(source, int) else str(source))
        if not cap.isOpened():
            self._set_status("error", f"Could not open the video: {source}")
            return False

        in_fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width > config.MAX_WIDTH:
            height = int(height * config.MAX_WIDTH / width)
            width = config.MAX_WIDTH
        _log(f"video opened: working size {width}x{height}, {in_fps:.1f} fps (round {loops + 1})")

        counter = VehicleCounter(line_ratio=config.LINE_RATIO)
        density = TrafficDensity()
        signal = SignalController(min_green=config.MIN_GREEN, max_green=config.MAX_GREEN,
                                  yellow=config.YELLOW, red=config.RED)
        violations = (ViolationDetector(flow=config.TRAFFIC_FLOW, line_ratio=config.LINE_RATIO,
                                        ignore_left=config.IGNORE_LEFT)
                      if config.ENABLE_VIOLATIONS else None)
        self._reset_models(detector, classifier, emergency)
        self._round = loops
        with self._lock:
            self._events.clear()

        name = Path(str(source)).name if not isinstance(source, int) else f"camera {source}"
        frame_idx = 0
        processed = 0
        smoothed_fps = 0.0
        last_save = 0.0                       # time of the last row stored in the database
        loop_start = time.time()              # the signal clock starts here
        was_emergency = False
        self._set_status("loading", "Processing the first frame ...")

        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if (frame_idx - 1) % config.FRAME_STRIDE:
                continue

            processed += 1
            t0 = time.time()

            def step(text):                   # progress lines for the first frames only (debugging)
                if processed <= DEBUG_FRAMES:
                    _log(f"frame {processed}: {text:<24} {time.time() - t0:6.2f} s")

            frame = cv2.resize(frame, (width, height))
            video_time = (frame_idx - 1) / in_fps

            # ---- the AI modules, one after the other ----
            detections = detector.track(frame)
            step("vehicle tracking done")
            if classifier:
                detections = classifier.update(frame, detections)
                step("type classifier done")
            counter.update(detections, height, video_time)
            dens = density.update(detections, frame.shape)

            confirmed = emergency.update(frame) if emergency else []
            step("emergency model done")
            real_emergency = any(d["priority"] for d in confirmed)
            emergency_active = real_emergency or self._simulated_emergency
            signal_time = (time.time() - loop_start) * config.SIGNAL_TIME_SCALE
            sig = signal.update(dens["status"], signal_time, emergency=emergency_active)

            new_violations = []
            if violations:
                new_violations = violations.update(
                    detections, frame.shape, video_time, density_status=dens["status"],
                    signal_phase=sig["phase"], check_red_light=config.CHECK_RED_LIGHT)

            # ---- alerts ----
            if emergency_active and not was_emergency:
                if real_emergency:
                    label = next(d["label"] for d in confirmed if d["priority"])
                    text = f"{label} detected - signal priority requested"
                else:
                    text = "Simulated emergency vehicle - signal priority requested"
                self._add_event(video_time, "emergency", text)
            elif was_emergency and not emergency_active:
                self._add_event(video_time, "info", "Emergency vehicle cleared")
            was_emergency = emergency_active

            for ev in new_violations:
                self._add_event(ev["time"], "violation",
                                f"{RULE_NAMES[ev['type']]}: {ev['label']} #{ev['id']}")
                
                        # ---- demo e-challans (simulation) ----
            if self._challans:
                for ev in new_violations:
                    self._challans.issue(frame, ev, "detected", loops)
                while self._sim_requests:
                    kind = self._sim_requests.popleft()
                    vehicle = self._challans.pick_vehicle(detections, kind)
                    if vehicle is None:
                        self._add_event(video_time, "info", "Demo violation skipped: no tracked vehicle in view")
                        continue
                    ev = {"type": kind, "id": vehicle["id"], "label": vehicle["label"],
                          "time": round(video_time, 1), "bbox": vehicle["bbox"]}
                    self._challans.issue(frame, ev, "simulated", loops)
                    self._add_event(video_time, "violation",
                                    f"SIMULATED (demo): {CHALLAN_TITLES[kind]} - {vehicle['label']} #{vehicle['id']}")

            # ---- the picture ----
            annotated = detector.draw(frame, detections)
            if config.OVERLAY_PANELS:
                counter.draw(annotated)
                density.draw(annotated, dens)
                signal.draw(annotated, sig)
                if violations:
                    violations.draw(annotated, video_time)
            else:
                self._draw_guides(annotated, density)      # only the counting line and the zone
                if violations:
                    violations.draw(annotated, video_time, show_bar=False)
            if emergency:
                emergency.draw(annotated, confirmed)
            if self._simulated_emergency:
                h = annotated.shape[0]
                cv2.rectangle(annotated, (0, h - 110), (annotated.shape[1], h - 72), (0, 0, 200), -1)
                cv2.putText(annotated, "EMERGENCY VEHICLE (simulated)", (8, h - 84),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            ok_jpg, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            step("picture ready")

            fps = 1.0 / max(time.time() - t0, 1e-6)
            smoothed_fps = fps if smoothed_fps == 0 else 0.9 * smoothed_fps + 0.1 * fps

            # ---- publish the results for the web routes ----
            with self._lock:
                if ok_jpg:
                    self._jpeg = buffer.tobytes()
                    self._frame_id += 1
                self._state.update({
                    "status": "running",
                    "message": "",
                    "video": {"name": name, "time": round(video_time, 1),
                              "fps": round(smoothed_fps, 1), "loops": loops},
                    "vehicles": counter.stats(),
                    "density": {"occupancy": round(dens["occupancy"], 3),
                                "smoothed": round(dens["smoothed"], 3),
                                "in_roi": dens["in_roi"], "status": dens["status"]},
                    "signal": {"phase": sig["phase"],
                               "time_left": round(max(0.0, sig["time_left"]), 1),
                               "phase_length": sig["phase_length"],
                               "next_green": sig["next_green"],
                               "priority_note": sig["priority_note"],
                               "time_scale": config.SIGNAL_TIME_SCALE},
                    "emergency": {"active": emergency_active,
                                  "simulated": self._simulated_emergency,
                                  "detected": [{"label": d["label"], "score": d["score"],
                                                "priority": d["priority"]} for d in confirmed]},
                    "violations": {"total": len(violations.events) if violations else 0,
                                   "by_type": violations.summary() if violations else {}},
                })

            # ---- store a row of statistics every few seconds ----
            if self._db and time.time() - last_save >= config.DB_SAVE_EVERY_S:
                last_save = time.time()
                self._db.add_stat(self.get_status())

        cap.release()
        return True