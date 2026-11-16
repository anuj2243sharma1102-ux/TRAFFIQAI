# Test results

Test date: 2026-11-13. Machine: Windows laptop, CPU only.

## 1. Automatic tests (`python tests\run_all_tests.py`)

| Test                                                         | Asserts | Result |
| ------------------------------------------------------------ | ------- | ------ |
| Traffic density: occupancy, status, hold time                | 9       | PASS   |
| Signal controller: cycle, adaptive green, emergency priority | 19      | PASS   |
| Violation rules: wrong way, stopped, red light               | 8       | PASS   |
| Database: store, summary, CSV export                         | 11      | PASS   |
| Vehicle counter: crossing, ID bridging, duplicate filter     | 21      | PASS   |

5 of 5 test files passed (68 assert statements). Full report: `output\test_report.txt`.

## 2. Vehicle counting (`cctv.mp4`, 32 s, one fixed camera)

Method: a picture was saved for every counted vehicle (`tests\audit_counting.py`) and
checked by eye for double counts or boxes on non-vehicles (precision). A review video
with a green box on every counted vehicle was watched to find vehicles that crossed the
line without a box (recall).

| Setting / change           | Counted | Note                                        |
| -------------------------- | ------- | ------------------------------------------- |
| yolov8n, 960 px, conf 0.35 | 54      | 1 double (precision 53/54 = 98%)            |
| 1280 px, conf 0.25         | 64      | 2 doubles, about 6 two-wheelers missed      |
| + ID bridging              | 67      | 3 missed vehicles recovered, 2 doubles left |
| + duplicate filter (final) | **65**  | 0 doubles, 3 known misses                   |

Final result: precision about 100% (65 of 65), recall about 96% (65 of 68 known
vehicles). Caution: the same clip was used for tuning and for measuring, and the missed
vehicles were found by one person watching the video.

Known misses: one two-wheeler that was already past the line when the video started, and
two scooters at about 17.2 s and 17.3 s that the detector never boxed.

## 3. Traffic density

### 3.1 Effect of the detector settings (`tests\run_density.py`, cctv.mp4, High >= 20%, ROI 15-55% of frame height)

|                                        | 960 px, conf 0.35, Low < 12% | 1280 px, conf 0.25, Low < 12% | 1280 px, conf 0.25, Low < 13% (final) |
| -------------------------------------- | ---------------------------- | ----------------------------- | ------------------------------------- |
| Vehicles crossed the line              | 54                           | 65                            | 65                                    |
| Speed (detection + tracking + density) | 5.5 FPS                      | 3.8 FPS                       | 3.6 FPS                               |
| Occupancy median                       | 16.9%                        | 17.2%                         | 17.2%                                 |
| Occupancy p10 / p90                    | 9.2% / 31.1%                 | 10.1% / 31.6%                 | 10.1% / 31.6%                         |
| Frames Low / Medium / High             | 36% / 20% / 43%              | 23% / 38% / 38%               | 31% / 29% / 41%                       |
| Status changes                         | 8                            | 10                            | 9                                     |

Smoothed occupancy per 5 s window (960 px / conf 0.35 -> 1280 px / conf 0.25):
0-5 s 11.1 -> 12.5%, 5-10 s 11.6 -> 12.4%, 10-15 s 14.1 -> 14.4%, 15-20 s 25.6 -> 25.2%,
20-25 s 22.0 -> 20.3%, 25-30 s 25.9 -> 28.8%, 30-35 s 22.8 -> 27.3%.

The occupancy itself changed little, but many frames lie near the 12% threshold, so the
Low/Medium mix moved. Final choice: Low < 13%, High >= 20% at 1280 px / conf 0.25. Raising
`low_max` by one point brought the mix back close to the first setting (31 / 29 / 41%
against 36 / 20 / 43%) with 9 status changes. The thresholds were chosen on this clip only,
so that all three states appear; there is no ground-truth congestion label.

### 3.2 Check by eye (12 moments of cctv.mp4)

Method: 12 fixed moments (1.5 s to 29.0 s, every 2.5 s) were chosen before looking at any
result. For each, a picture with only the region of interest drawn was saved
(`tests\density_check_frames.py`). One person wrote Low / Medium / High for every picture
before seeing the program's answer; the two were then compared
(`tests\density_check_compare.py`).

| #   | Time   | Person | Program | Smoothed occupancy | Result     |
| --- | ------ | ------ | ------- | ------------------ | ---------- |
| 1   | 1.5 s  | Low    | Medium  | 11.5%              | off by one |
| 2   | 4.0 s  | Low    | Low     | 15.9%              | match      |
| 3   | 6.5 s  | Low    | Medium  | 11.9%              | off by one |
| 4   | 9.0 s  | Low    | Low     | 11.5%              | match      |
| 5   | 11.5 s | Low    | Low     | 13.9%              | match      |
| 6   | 14.0 s | Medium | Medium  | 14.2%              | match      |
| 7   | 16.5 s | High   | High    | 32.3%              | match      |
| 8   | 19.0 s | Medium | Medium  | 17.8%              | match      |
| 9   | 21.5 s | Medium | High    | 22.9%              | off by one |
| 10  | 24.0 s | High   | Low     | 9.5%               | off by two |
| 11  | 26.5 s | High   | High    | 25.9%              | match      |
| 12  | 29.0 s | High   | High    | 30.4%              | match      |

Exact match: 8 of 12 (67%). Within one level: 11 of 12 (92%).

| Person says | Program Low | Program Medium | Program High |
| ----------- | ----------- | -------------- | ------------ |
| Low         | 3           | 2              | 0            |
| Medium      | 0           | 2              | 1            |
| High        | 1           | 0              | 3            |

Observations:

- The program's status follows the occupancy with a delay: occupancy is smoothed and a new
  status must last 30 frames (1 s) before it is shown. In pictures 1 and 3 the status is
  Medium although the occupancy is below the 13% threshold, probably because of this delay
  and the state at the start of the video.
- Picture 9 (22.9%) lies close to the 20% threshold.
- Picture 10 is the only large disagreement. A large bus is in the middle of the region of
  interest, but the occupancy is only 9.5%. Probably the detector does not box the roof of
  the bus until its red rear becomes visible, and the smoothing delays the rise (not
  verified).
- Caution: one person labelled, only 12 samples, and the thresholds were tuned on the same
  clip, so this is an estimate, not a precise accuracy.

## 4. Emergency vehicle detection (`tests\evaluate_emergency.py`)

- 3 of 3 checks pass.
- Convoy flagged in 212 of 216 frames of the held-out part of `amb_fire.mp4`.
- 0 false alarms in 955 normal frames of `cctv.mp4`.

## 5. Vehicle type classifier

- 94% on 109 held-out crops. Weak classes: truck (6 crops), taxi (5 crops).

## 6. Speed and frame skipping (`python tests\benchmark_stride.py 1,2,3,4,6`)

Settings: 1280 px, conf 0.25, all AI modules, without the web server and the database.
Video: 954 frames = 31.8 s at 30 fps.

| FRAME_STRIDE | Frames used | Time (s) | Speed vs real time | Vehicles counted | Difference | False violations |
| ------------ | ----------- | -------- | ------------------ | ---------------- | ---------- | ---------------- |
| 1            | 954         | 322      | 0.10x              | 65               | 0          | 0                |
| 2            | 477         | 168      | 0.19x              | 42               | -23        | 2                |
| 3            | 318         | 112      | 0.28x              | 40               | -25        | 4                |
| 4            | 239         | 105      | 0.30x              | 26               | -39        | 5                |
| 6            | 159         | 52       | 0.61x              | 13               | -52        | 4                |

`cctv.mp4` contains no real violation, so every violation reported at stride 2 or more is
a false alarm.

Vehicle types counted:

| FRAME_STRIDE | car | motorcycle | tempo | auto | bus | truck |
| ------------ | --- | ---------- | ----- | ---- | --- | ----- |
| 1            | 14  | 30         | 6     | 11   | 2   | 2     |
| 2            | 12  | 11         | 6     | 10   | 2   | 1     |
| 3            | 12  | 9          | 5     | 10   | 2   | 2     |
| 4            | 11  | 2          | 4     | 7    | 1   | 1     |
| 6            | 5   | 1          | 3     | 1    | 0   | 3     |

Observations:

- Skipping frames loses vehicles, and motorcycles most of all: at stride 2, 19 of the 23
  lost vehicles are motorcycles (30 -> 11). Probably small, fast vehicles move too far
  between two processed frames for the tracker to follow them (not verified).
- Skipping frames also creates false violations.
- The time of stride 4 (105 s) looks too high compared with stride 3 (112 s); other load on
  the laptop may have disturbed it, so it should not be over-interpreted.
- An earlier run at 960 px / conf 0.35 (before the counter fixes) showed the same trend
  (54 vehicles at stride 1, 42 at stride 2).
- Chosen value: `FRAME_STRIDE = 1`. At this value the system runs at about 0.10x real
  time (about 3 FPS). Even stride 6 (0.61x) is slower than real time, so real-time use on
  this CPU would need a smaller image size, a faster detector or a GPU.

Full pipeline on the dashboard (1280 px, conf 0.25): about 3.1 FPS (954 frames in about
305 s), consistent with the benchmark at stride 1 (about 3.0 FPS).

## 7. System check

- `python tests\check_system.py`: 19 checks pass.

## 8. Second clip (`traffic.mp4`, moving drone)

- Not usable for validation: the camera moves and the vehicles are tiny. The system counted
  only 23 vehicles in 18 s and most vehicles had no box. See LIMITATIONS.md.
