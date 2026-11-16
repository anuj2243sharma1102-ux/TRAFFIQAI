"""All settings of the system in one place."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# ---- input ----
VIDEO_SOURCE = PROJECT_ROOT / "videos" / "cctv.mp4"   # or 0 for a webcam
LOOP_VIDEO = True            # start again when the video ends
MAX_WIDTH = 1280             # bigger frames are shrunk to this width
FRAME_STRIDE = 1             # 1 = every frame, 2 = every second frame (faster, less exact)

# ---- models ----
VEHICLE_MODEL = PROJECT_ROOT / "models" / "yolov8n.pt"
EMERGENCY_MODEL = PROJECT_ROOT / "models" / "emergency2.pt"
VEHICLE_TYPE_MODEL = PROJECT_ROOT / "models" / "vehicle_type.pt"
IMAGE_SIZE = 1280
CONFIDENCE = 0.25

# ---- switches ----
ENABLE_TYPES = True          # vehicle type classifier (auto, tempo, taxi ...)
ENABLE_EMERGENCY = True      # emergency vehicle detector
ENABLE_VIOLATIONS = True     # violation rules
CHECK_RED_LIGHT = False      # needs the REAL signal state, so off for our simulated signal
OVERLAY_PANELS = False       # clean video: the numbers are on the dashboard        # draw the info panels on the video
EMERGENCY_EVERY_N = 3        # run the emergency model on every 3rd frame (saves time)

# ---- traffic logic ----
LINE_RATIO = 0.4             # counting / stop line at 40% of the frame height
TRAFFIC_FLOW = "up"          # normal direction of traffic in the picture: "up" or "down"
IGNORE_LEFT = 0.3            # ignore "stopped" in the left 30% (taxis at the kerb)
MIN_GREEN, MAX_GREEN, YELLOW, RED = 15, 45, 3, 20
SIGNAL_TIME_SCALE = 1.0      # 1 = the signal runs in real seconds (3.0 = 3x faster, for demos)      # demo: the signal clock runs 3x faster than the video

# ---- web server ----
HOST = "127.0.0.1"
PORT = 5000

# ---- database ----
DB_PATH = PROJECT_ROOT / "output" / "traffic.db"
DB_SAVE_EVERY_S = 5          # store one row of statistics every 5 seconds
DEBUG_FIRST_FRAMES = 0     # a number > 0 prints the progress of the first frames (debugging)
# ---- demo e-challan (simulation only: no legal value, no real vehicle data) ----
ENABLE_CHALLAN_DEMO = True
CHALLAN_FOLDER = PROJECT_ROOT / "output" / "challans"
DEMO_CAMERA_ID = "CAM-01"
DEMO_LOCATION = "Demo junction (recorded video)"
DEMO_FINES = {                       # SAMPLE amounts in rupees, not the legal fines
    "wrong_way": 100, "stopped": 100, "red_light": 100,
    "over_speed": 100, "no_helmet": 100, "seat_belt": 100, "mobile_phone": 100,
}