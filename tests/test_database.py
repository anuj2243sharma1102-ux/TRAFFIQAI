"""Check the database module with made-up data (no video, no server)."""

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.database import Database


def state(total, live, status, occupancy, loops=0, emergency=False):
    return {
        "video": {"name": "t.mp4", "time": float(total), "fps": 4.5, "loops": loops},
        "vehicles": {"live": live, "smoothed": live, "total": total, "down": 0, "up": total,
                     "by_type": {"car": total}},
        "density": {"occupancy": occupancy, "smoothed": occupancy, "in_roi": live, "status": status},
        "signal": {"phase": "GREEN", "time_left": 12.3, "phase_length": 15, "next_green": 15,
                   "priority_note": "", "time_scale": 1.0},
        "emergency": {"active": emergency, "simulated": emergency, "detected": []},
        "violations": {"total": 0, "by_type": {}},
    }


with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
    db = Database(Path(folder) / "test.db")
    db.add_stat(state(2, 5, "Low", 0.06))
    db.add_stat(state(6, 9, "Medium", 0.15))
    db.add_stat(state(10, 14, "High", 0.30))
    db.add_stat(state(3, 4, "Low", 0.05, loops=1))        # a second round of the video
    db.add_event({"video_time": 4.2, "type": "emergency", "text": "Simulated emergency"}, 0)
    db.add_event({"video_time": 9.0, "type": "violation", "text": "Stopped vehicle: car #7"}, 0)

    summary = db.summary()
    print("summary:", summary)
    assert summary["samples"] == 4
    assert summary["vehicles_counted"] == 13              # round 0 ended at 10, round 1 at 3
    assert summary["peak_in_view"] == 14
    assert abs(summary["avg_occupancy"] - 0.14) < 0.001
    assert abs(summary["status_share"]["Low"] - 0.5) < 0.001
    assert summary["events"] == {"emergency": 1, "violation": 1}

    rows = db.recent_stats(limit=3)
    print("last 3 rows, oldest first:", [r["vehicles_total"] for r in rows])
    assert [r["vehicles_total"] for r in rows] == [6, 10, 3]
    assert rows[-1]["by_type"] == {"car": 3}

    events = db.recent_events(limit=10)
    assert events[0]["type"] == "violation"               # newest first

    stats_csv = db.export_csv("traffic_stats").strip().splitlines()
    events_csv = db.export_csv("events").strip().splitlines()
    print("csv header:", stats_csv[0])
    assert len(stats_csv) == 5 and len(events_csv) == 3
    assert stats_csv[0].startswith("id,ts,session")

    db.close()

print("All database checks passed.")