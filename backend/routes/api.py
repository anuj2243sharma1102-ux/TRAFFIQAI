"""Web routes: they only read the latest results from the pipeline or the database."""

import time

from flask import Blueprint, Response, current_app, jsonify, request

api = Blueprint("api", __name__)


def _pipeline():
    return current_app.extensions["pipeline"]


def _database():
    return current_app.extensions["database"]


@api.get("/api/health")
def health():
    return jsonify({"ok": True})


@api.get("/api/status")
def status():
    return jsonify(_pipeline().get_status())


@api.get("/api/events")
def events():
    limit = request.args.get("limit", 20, type=int)
    return jsonify(_pipeline().get_events(limit))


@api.post("/api/emergency")
def emergency():
    """Simulate (or clear) an emergency vehicle. Body: {"active": true}"""
    data = request.get_json(silent=True) or {}
    _pipeline().set_simulated_emergency(bool(data.get("active", True)))
    return jsonify(_pipeline().get_status()["emergency"])


# ---------------------------------------------------------------- stored history (SQLite)
@api.get("/api/history")
def history():
    limit = min(request.args.get("limit", 200, type=int), 5000)
    minutes = request.args.get("minutes", None, type=int)
    return jsonify(_database().recent_stats(limit=limit, minutes=minutes))


@api.get("/api/history/events")
def history_events():
    limit = min(request.args.get("limit", 50, type=int), 1000)
    return jsonify(_database().recent_events(limit=limit))


@api.get("/api/summary")
def summary():
    return jsonify(_database().summary())


def _csv_download(table, filename):
    return Response(_database().export_csv(table), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={filename}"})


@api.get("/api/export/stats.csv")
def export_stats():
    return _csv_download("traffic_stats", "traffic_stats.csv")


@api.get("/api/export/events.csv")
def export_events():
    return _csv_download("events", "traffic_alerts.csv")


# ---------------------------------------------------------------- live video
def _mjpeg(pipeline):
    """Send the newest picture again and again (Motion JPEG stream)."""
    last_id = -1
    while True:
        frame_id, jpeg = pipeline.get_frame()
        if jpeg is None or frame_id == last_id:
            time.sleep(0.03)
            continue
        last_id = frame_id
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"


@api.get("/video_feed")
def video_feed():
    return Response(_mjpeg(_pipeline()),
                    mimetype="multipart/x-mixed-replace; boundary=frame")