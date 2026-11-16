"""SQLite storage: traffic statistics (one row every few seconds) and alerts.

Every write is wrapped in try/except: if the database is busy (for example OneDrive is
syncing the file) the AI pipeline must not crash, we just skip that row.
"""

import csv
import io
import json
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS traffic_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    session TEXT,
    round INTEGER,
    video_time REAL,
    vehicles_total INTEGER,
    vehicles_live INTEGER,
    by_type TEXT,
    occupancy REAL,
    density_status TEXT,
    signal_phase TEXT,
    signal_time_left REAL,
    next_green INTEGER,
    emergency_active INTEGER,
    violations_total INTEGER,
    fps REAL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    session TEXT,
    round INTEGER,
    video_time REAL,
    type TEXT,
    text TEXT
);
CREATE TABLE IF NOT EXISTS challans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    challan_no TEXT,
    ts TEXT NOT NULL,
    session TEXT,
    round INTEGER,
    video_time REAL,
    violation TEXT,
    source TEXT,
    vehicle_id INTEGER,
    vehicle_class TEXT,
    plate TEXT,
    camera TEXT,
    location TEXT,
    fine INTEGER,
    evidence_file TEXT,
    pdf_file TEXT
);
CREATE INDEX IF NOT EXISTS idx_stats_ts ON traffic_stats(ts);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
"""


def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


class Database:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.session = _now()                 # one value for every start of the program
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), timeout=10, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def close(self):
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------ small helpers
    def _write(self, sql, params):
        try:
            with self._lock:
                self._conn.execute(sql, params)
                self._conn.commit()
            return True
        except sqlite3.Error as exc:
            print(f"[database] could not write: {exc}", flush=True)
            return False

    def _read(self, sql, params=()):
        try:
            with self._lock:
                return [dict(row) for row in self._conn.execute(sql, params).fetchall()]
        except sqlite3.Error as exc:
            print(f"[database] could not read: {exc}", flush=True)
            return []

    # ------------------------------------------------------------ writing
    def add_stat(self, state):
        """Store one row from the pipeline status dictionary."""
        v, d, s = state["vehicles"], state["density"], state["signal"]
        e, vio, vid = state["emergency"], state["violations"], state["video"]
        return self._write(
            "INSERT INTO traffic_stats (ts, session, round, video_time, vehicles_total, "
            "vehicles_live, by_type, occupancy, density_status, signal_phase, signal_time_left, "
            "next_green, emergency_active, violations_total, fps) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (_now(), self.session, vid["loops"], vid["time"], v["total"], v["live"],
             json.dumps(v["by_type"]), d["smoothed"], d["status"], s["phase"],
             s["time_left"], s["next_green"], int(e["active"]), vio["total"], vid["fps"]))

    def add_event(self, event, round_no=0):
        """Store one alert (emergency, violation ...)."""
        return self._write(
            "INSERT INTO events (ts, session, round, video_time, type, text) VALUES (?,?,?,?,?,?)",
            (_now(), self.session, round_no, event["video_time"], event["type"], event["text"]))

    def add_challan(self, record):
        """Store one DEMO challan and return its number (DEMO-00001, DEMO-00002 ...)."""
        with self._lock:
            cursor = self._conn.execute(
                "INSERT INTO challans (ts, session, round, video_time, violation, source, "
                "vehicle_id, vehicle_class, plate, camera, location, fine) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (record["ts"], record["session"], record["round"], record["video_time"],
                 record["violation"], record["source"], record["vehicle_id"],
                 record["vehicle_class"], record["plate"], record["camera"],
                 record["location"], record["fine"]))
            number = f"DEMO-{cursor.lastrowid:05d}"
            self._conn.execute("UPDATE challans SET challan_no = ? WHERE id = ?",
                               (number, cursor.lastrowid))
            self._conn.commit()
        return number

    def set_challan_files(self, number, evidence_file, pdf_file):
        """Remember the names of the evidence picture and the PDF of a challan."""
        return self._write("UPDATE challans SET evidence_file = ?, pdf_file = ? WHERE challan_no = ?",
                           (evidence_file, pdf_file, number))

    # ------------------------------------------------------------ reading
    def recent_stats(self, limit=200, minutes=None):
        """Latest rows, oldest first."""
        if minutes:
            cutoff = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time() - minutes * 60))
            rows = self._read("SELECT * FROM traffic_stats WHERE ts >= ? ORDER BY id DESC LIMIT ?",
                              (cutoff, limit))
        else:
            rows = self._read("SELECT * FROM traffic_stats ORDER BY id DESC LIMIT ?", (limit,))
        rows.reverse()
        for row in rows:
            row["by_type"] = json.loads(row["by_type"] or "{}")
        return rows

    def recent_events(self, limit=50):
        """Latest alerts, newest first."""
        return self._read("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,))

    def recent_challans(self, limit=10):
        """Latest demo challans, newest first."""
        return self._read("SELECT * FROM challans ORDER BY id DESC LIMIT ?", (limit,))

    def summary(self):
        s = (self._read(
            "SELECT COUNT(*) AS n, MIN(ts) AS first_ts, MAX(ts) AS last_ts, AVG(occupancy) AS avg_occ, "
            "MAX(occupancy) AS peak_occ, AVG(vehicles_live) AS avg_live, MAX(vehicles_live) AS peak_live "
            "FROM traffic_stats") or [{}])[0]
        n = s.get("n") or 0
        shares = {r["density_status"]: r["c"] for r in self._read(
            "SELECT density_status, COUNT(*) AS c FROM traffic_stats GROUP BY density_status")}
        # the counter starts again at 0 in every round of the video, so add up the last value of each round
        counted = self._read(
            "SELECT COALESCE(SUM(m), 0) AS total FROM "
            "(SELECT MAX(vehicles_total) AS m FROM traffic_stats GROUP BY session, round)")
        events = {r["type"]: r["c"] for r in self._read(
            "SELECT type, COUNT(*) AS c FROM events GROUP BY type")}
        return {
            "samples": n,
            "first": s.get("first_ts"),
            "last": s.get("last_ts"),
            "avg_occupancy": round(s.get("avg_occ") or 0, 3),
            "peak_occupancy": round(s.get("peak_occ") or 0, 3),
            "avg_in_view": round(s.get("avg_live") or 0, 1),
            "peak_in_view": s.get("peak_live") or 0,
            "status_share": {k: round(v / n, 3) for k, v in shares.items()} if n else {},
            "vehicles_counted": counted[0]["total"] if counted else 0,
            "events": events,
        }

    def export_csv(self, table):
        """Whole table as CSV text (for Excel / the report)."""
        if table not in ("traffic_stats", "events", "challans"):
            raise ValueError(f"unknown table: {table}")
        with self._lock:
            cursor = self._conn.execute(f"SELECT * FROM {table} ORDER BY id")
            columns = [c[0] for c in cursor.description]
            rows = cursor.fetchall()
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(columns)
        writer.writerows([tuple(row) for row in rows])
        return buffer.getvalue()