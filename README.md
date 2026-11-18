# TRAFFIQ AI - AI-Based Smart Traffic Monitoring And Management System

A college PBL project: a computer-vision prototype that watches a fixed traffic camera video,
counts vehicles, measures how full the road is, adapts a simulated traffic signal, gives priority
to emergency vehicles, flags a few violations and shows everything on a live web dashboard.
It runs on a normal laptop CPU.

> **Prototype for learning and demonstration.** The signal is simulated and the e-challan feature
> is a demo with sample plates and sample amounts. Nothing here issues a legal challan or controls
> a real traffic light.

## Features

- Vehicle detection and tracking with YOLOv8n + ByteTrack (car, motorcycle, bus, truck)
- Vehicle type classifier for Indian traffic (auto-rickshaw, tempo, taxi ...)
- Counting at a virtual line, with track-ID bridging and a duplicate filter
- Traffic density (Low / Medium / High) from the share of the road covered by vehicles
- Adaptive signal timing (simulation): green time 15 / 30 / 45 s depending on density
- Emergency vehicle detection (ambulance, fire engine) with signal priority, plus a simulated trigger
- Violations: wrong way and stopped vehicle (red light needs a real signal state and is off by default)
- Demo e-challan: evidence picture with camera, place, date and time, sample PDF, table on the dashboard
  (marked DETECTED or SIMULATED)
- SQLite history of statistics and alerts, CSV export
- Web dashboard (Flask + HTML/CSS/JavaScript)

## Screenshots

![Dashboard](docs/images/dashboard.png)
![Demo e-challans and history](docs/images/challans_and_history.png)
![Sample challan PDF](docs/images/sample_challan.png)

## Results (measured on the 32 s test clip `cctv.mp4`)

| Part                                           | Result                                                                     |
| ---------------------------------------------- | -------------------------------------------------------------------------- |
| Vehicle counting                               | 65 counted, precision about 100%, recall about 96% (3 two-wheelers missed) |
| Traffic density, check by eye on 12 moments    | 8 of 12 exact, 11 of 12 within one level                                   |
| Emergency vehicles (held-out part of the clip) | convoy flagged in 212 of 216 frames, 0 false alarms in 955 normal frames   |
| Vehicle type classifier                        | 94% on 109 held-out crops (truck and taxi are weak)                        |
| Automatic tests                                | 6 test files, 83 assert statements, all pass                               |
| Speed (CPU only)                               | about 3 FPS, roughly 10 times slower than real time                        |

The counting numbers come from the same clip that was used for tuning, so they are not an
independent validation. Details: [docs/TEST_RESULTS.md](docs/TEST_RESULTS.md) and
[docs/LIMITATIONS.md](docs/LIMITATIONS.md).

## Tech stack

Python, Ultralytics YOLO, OpenCV, NumPy, PyTorch (CPU), Flask, SQLite, ReportLab,
HTML / CSS / JavaScript

## Project structure

- `backend/` - Flask server: `app.py`, `routes/` (API, demo challans), `services/` (pipeline, database, demo challan)
- `detection/` - AI and computer-vision modules: detector, counter, type classifier, density,
  signal controller, emergency detector, violation detector
- `frontend/` - dashboard (`index.html`, `css/`, `js/`)
- `tests/` - automatic tests and evaluation scripts
- `docs/` - test results and limitations
- `models/`, `videos/`, `output/` - model weights, input videos, results (database, challans)
- `config.py` - all settings in one place; `app.py` - starts the server; `tests/check_system.py` - setup check

## Setup (Windows)

Tested with Python 3.13.7 on Windows.

```
git clone https://github.com/anuj2243sharma1102-ux/TRAFFIQAI.git
cd TRAFFIQAI
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

1. Put a fixed-camera traffic video at `videos\cctv.mp4` (or change `VIDEO_SOURCE` in `config.py`).
   Videos are not included in this repository.
2. 2. `emergency2.pt` and `vehicle_type.pt` (trained for this project) are included in `models\`.
      Download `yolov8n.pt` from Ultralytics and put it in `models\` as well.
3. Check the setup: `python tests\check_system.py`

## Run

```
python app.py
```

Open http://127.0.0.1:5000 in the browser. The video plays slower than real time because
every frame is analysed on the CPU.

## Tests

```
python tests\run_all_tests.py
```

## Demo e-challan

The dashboard has a panel "Demo e-challans". Wrong-way and stopped vehicles are detected by the
program. Red light, over-speeding, no helmet, no seat belt and mobile phone use are only triggered
with the "Simulate violation" button and are marked SIMULATED. Plates are sample plates (`DEMO-0007`)
and amounts are sample amounts (`DEMO_FINES` in `config.py`). The PDFs carry a DEMO watermark.

## Limitations (short)

- Needs a fixed camera; it does not work on moving (drone) footage.
- The nano YOLO model misses some small two-wheelers.
- The signal is simulated; red-light checking needs a real signal state.
- Counting thresholds were tuned on one clip.
- The emergency model was trained on a small dataset.

## Credits

Ultralytics YOLOv8, ByteTrack, OpenCV, Flask, ReportLab.
Test video: footage by Javed Shaikh on Pexels (https://www.pexels.com), used under the Pexels License.
The video is not included in this repository.
