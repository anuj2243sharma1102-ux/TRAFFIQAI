"""Check the demo e-challan feature with made-up data (no video, no model, no server)."""

import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from backend.services.challan_demo import ChallanDemo, SIMULATED_TYPES, TITLES
from backend.services.database import Database


def det(vid, label, bbox):
    return {"id": vid, "label": label, "bbox": bbox}


with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
    config.CHALLAN_FOLDER = Path(folder) / "challans"          # keep the test files away from output\
    db = Database(Path(folder) / "test.db")
    demo = ChallanDemo(db)

    # ---- which vehicle is used for a simulated violation ----
    detections = [det(1, "car", (0, 0, 200, 200)), det(2, "motorcycle", (0, 0, 50, 60)),
                  det(3, "bus", (0, 0, 300, 400)), det(None, "car", (0, 0, 900, 900))]
    assert demo.pick_vehicle(detections, "over_speed")["id"] == 3      # biggest tracked box
    assert demo.pick_vehicle(detections, "no_helmet")["id"] == 2       # a helmet fits a motorcycle
    assert demo.pick_vehicle(detections, "seat_belt")["id"] == 1       # a seat belt fits a car
    assert demo.pick_vehicle([det(None, "car", (0, 0, 9, 9))], "over_speed") is None
    assert demo.pick_vehicle([], "over_speed") is None
    assert set(SIMULATED_TYPES) <= set(TITLES)

    # ---- one demo challan: database row, evidence picture, PDF ----
    frame = np.full((480, 640, 3), 90, np.uint8)
    event = {"type": "wrong_way", "id": 7, "label": "car", "time": 12.3, "bbox": (100, 100, 220, 200)}
    demo._make(frame, event, "detected", 0)                            # directly, not in a thread

    rows = db.recent_challans(limit=5)
    print("challan row:", rows[0])
    assert len(rows) == 1
    row = rows[0]
    assert row["challan_no"] == "DEMO-00001" and row["plate"] == "DEMO-0007"
    assert row["source"] == "detected" and row["violation"] == "wrong_way"
    assert row["fine"] == config.DEMO_FINES["wrong_way"]

    evidence = Path(config.CHALLAN_FOLDER) / row["evidence_file"]
    pdf = Path(config.CHALLAN_FOLDER) / row["pdf_file"]
    assert evidence.exists() and evidence.stat().st_size > 1000
    assert pdf.exists() and pdf.read_bytes().startswith(b"%PDF")

    # ---- a second one gets the next number; a simulated one is marked as simulated ----
    demo._make(frame, {"type": "no_helmet", "id": 12, "label": "motorcycle", "time": 3.0,
                       "bbox": (10, 10, 60, 90)}, "simulated", 0)
    rows = db.recent_challans(limit=5)
    assert [r["challan_no"] for r in rows] == ["DEMO-00002", "DEMO-00001"]     # newest first
    assert rows[0]["source"] == "simulated"

    # ---- CSV export ----
    lines = db.export_csv("challans").strip().splitlines()
    assert len(lines) == 3 and lines[0].startswith("id,challan_no")

    db.close()

print("All demo challan checks passed.")