"""Demo e-challan: an evidence picture, a sample PDF and a row in the database.

THIS IS A SIMULATION FOR A STUDENT PROJECT. It has no legal value, uses sample plates
(DEMO-0001 ...) and sample amounts, and is not connected to any real database.

Violations the program really detects (wrong way, stopped) and violations that are only
triggered with the dashboard button (red light, speed, helmet ...) both end up here;
the column "source" tells which is which (detected / simulated).
"""

import threading
import time
from pathlib import Path

import cv2
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

import config

TITLES = {
    "wrong_way": "Wrong-way driving",
    "stopped": "Vehicle stopped on the carriageway",
    "red_light": "Red-light violation",
    "over_speed": "Over-speeding",
    "no_helmet": "Riding without a helmet",
    "seat_belt": "Not wearing a seat belt",
    "mobile_phone": "Using a mobile phone while driving",
}

# the types the dashboard button can trigger (the program does not detect these on this video)
SIMULATED_TYPES = ("red_light", "over_speed", "no_helmet", "seat_belt", "mobile_phone")

# vehicle types that fit a simulated violation best (otherwise any tracked vehicle is used)
PREFERRED = {
    "no_helmet": ("motorcycle",),
    "seat_belt": ("car", "taxi"),
    "mobile_phone": ("car", "taxi", "motorcycle"),
}

HOW = {"detected": "Detected by the program", "simulated": "Simulated trigger (demo)"}


def _evidence_picture(frame, bbox, stamp):
    """The frame with a red box around the vehicle and a stamp with camera, place, date and time."""
    picture = frame.copy()
    height, width = picture.shape[:2]
    x1, y1, x2, y2 = bbox
    cv2.rectangle(picture, (x1, y1), (x2, y2), (0, 0, 255), 4)
    cv2.rectangle(picture, (0, height - 60), (width, height), (0, 0, 0), -1)
    cv2.putText(picture, stamp, (10, height - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(picture, "DEMO - SIMULATION", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
    return picture


def _make_pdf(record, evidence_path, pdf_path):
    """A one-page sample challan with the evidence picture and a DEMO watermark."""
    width, height = A4
    pdf = canvas.Canvas(str(pdf_path), pagesize=A4)
    pdf.setTitle(f"Sample challan {record['challan_no']} (DEMO)")

    # header
    pdf.setFillColorRGB(0.12, 0.16, 0.23)
    pdf.rect(0, height - 80, width, 80, stroke=0, fill=1)
    pdf.setFillColorRGB(1, 1, 1)
    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(40, height - 42, "SAMPLE TRAFFIC CHALLAN (DEMO)")
    pdf.setFont("Helvetica", 11)
    pdf.drawString(40, height - 62, "Student project simulation - not a legal document")

    # details
    rows = [
        ("Challan no.", record["challan_no"]),
        ("Date and time", record["ts"]),
        ("Camera", record["camera"]),
        ("Location", record["location"]),
        ("Video time", f"{record['video_time']:.1f} s"),
        ("Sample plate", record["plate"]),
        ("Vehicle type (AI)", record["vehicle_class"]),
        ("Violation", TITLES.get(record["violation"], record["violation"])),
        ("How it was found", HOW.get(record["source"], record["source"])),
        ("Sample amount", f"Rs {record['fine']} (sample, not the legal fine)"),
    ]
    y = height - 118
    for label, value in rows:
        pdf.setFillColorRGB(0.3, 0.35, 0.42)
        pdf.setFont("Helvetica-Bold", 11)
        pdf.drawString(40, y, label)
        pdf.setFillColorRGB(0, 0, 0)
        pdf.setFont("Helvetica", 12)
        pdf.drawString(190, y, str(value))
        y -= 22

    # evidence picture, as large as the free space allows
    pdf.setFillColorRGB(0.3, 0.35, 0.42)
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(40, y - 8, "Evidence picture (stamped with camera, place, date and time)")
    image = ImageReader(str(evidence_path))
    image_w, image_h = image.getSize()
    scale = min((width - 80) / image_w, (y - 100) / image_h)
    draw_w, draw_h = image_w * scale, image_h * scale
    pdf.drawImage(image, (width - draw_w) / 2, y - 20 - draw_h, draw_w, draw_h)

    # footer
    pdf.setFillColorRGB(0.4, 0.4, 0.4)
    pdf.setFont("Helvetica", 9)
    pdf.drawString(40, 40, "Sample plate and sample amount. No real vehicle or owner data was used.")
    pdf.drawString(40, 28, "Created by the AI-Based Smart Traffic Monitoring System (demo).")

    # diagonal watermark on top of everything
    pdf.saveState()
    pdf.setFillColorRGB(0.85, 0.1, 0.1)
    pdf.setFillAlpha(0.18)
    pdf.setFont("Helvetica-Bold", 40)
    pdf.translate(width / 2, height / 2)
    pdf.rotate(35)
    pdf.drawCentredString(0, 0, "DEMO - NOT A LEGAL CHALLAN")
    pdf.restoreState()

    pdf.showPage()
    pdf.save()


class ChallanDemo:
    def __init__(self, database):
        self.db = database
        self.folder = Path(config.CHALLAN_FOLDER)
        self.folder.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def pick_vehicle(detections, kind):
        """The tracked vehicle (with an ID) that fits this violation best: the biggest box."""
        tracked = [d for d in detections if d["id"] is not None]
        preferred = [d for d in tracked if d["label"] in PREFERRED.get(kind, ())]
        pool = preferred or tracked
        if not pool:
            return None
        return max(pool, key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]))

    def issue(self, frame, event, source, round_no=0):
        """Make the demo challan in a background thread, so the video does not stall.

        event = {"type", "id", "label", "time", "bbox"}; source = "detected" or "simulated".
        """
        threading.Thread(target=self._make, args=(frame.copy(), dict(event), source, round_no),
                         daemon=True, name="challan").start()

    def _make(self, frame, event, source, round_no):
        try:
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            record = {
                "ts": now, "session": self.db.session, "round": round_no,
                "video_time": event["time"], "violation": event["type"], "source": source,
                "vehicle_id": event["id"], "vehicle_class": event["label"],
                "plate": f"DEMO-{event['id']:04d}",           # a sample plate, never read from the video
                "camera": config.DEMO_CAMERA_ID, "location": config.DEMO_LOCATION,
                "fine": config.DEMO_FINES.get(event["type"], 0),
            }
            record["challan_no"] = self.db.add_challan(record)
            stamp = f"{record['camera']} | {record['location']} | {now} | video {record['video_time']:.1f} s"
            evidence_name = record["challan_no"] + ".jpg"
            pdf_name = record["challan_no"] + ".pdf"
            picture = _evidence_picture(frame, event["bbox"], stamp)
            cv2.imwrite(str(self.folder / evidence_name), picture, [cv2.IMWRITE_JPEG_QUALITY, 85])
            _make_pdf(record, self.folder / evidence_name, self.folder / pdf_name)
            self.db.set_challan_files(record["challan_no"], evidence_name, pdf_name)
            print(f"[challan] {record['challan_no']} created ({source} {event['type']})", flush=True)
        except Exception as exc:
            print(f"[challan] could not create the demo challan: {exc}", flush=True)