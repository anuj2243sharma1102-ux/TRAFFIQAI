# Limitations

These are the known limits of the system, measured or observed during testing.

## Counting

- Tested and tuned on one clip (`cctv.mp4`, 32 s, fixed camera). The numbers below
  are not an independent validation.
- `yolov8n` (nano model) sometimes does not detect small two-wheelers. On the test clip,
  about 3 two-wheelers were missed: 2 were never detected, 1 appeared past the counting
  line at the start of the video.
- A vehicle that is first detected after the counting line (for example at the very
  start of the video) is not counted.
- The ID bridging and duplicate filter use pixel distances tuned for this clip. On a
  clip with tiny, closely spaced vehicles (a drone video of a dense highway) they merge
  separate vehicles, so the system is not meant for that kind of footage.
- The counter assumes a **fixed camera**. A moving camera (drone) makes line crossing
  meaningless.

## Speed

- The system runs on CPU only. The full pipeline is much slower than real time
  (video plays several times slower).
- Skipping frames to speed it up loses vehicles, mainly motorcycles (65 vehicles at stride 1, 42 at stride 2, 13 at stride 6) and creates false violations, so `FRAME_STRIDE` stays 1. Even stride 6 is slower than real time (0.61x).

## Traffic density and signal

- One large nearby bus can cover a big part of the region of interest and push the
  status to High.
- Density thresholds (Low < 13%, High >= 20%) were chosen on this clip only.
- The signal controller is a simulation. It is not connected to a real traffic light,
- The status follows the occupancy with a delay of about 1 s (smoothing plus hold time), so it can lag behind a sudden change. In a check of 12 moments by one person, the program matched exactly in 8 and within one level in 11.
- A large bus seen from above may not be boxed until its rear becomes visible, so the occupancy can be too low when a bus enters the region of interest.
  so red-light violation checking is switched off.
  “Changing the detector settings (960/0.35 to 1280/0.25) moved the share of Low frames from 36% to 23% although the median occupancy hardly changed; frames near a threshold change status easily.”

## Emergency vehicle detection

- The model (`emergency2.pt`) was trained on a very small set of own frames
  (28 with an emergency vehicle, 39 without). The validation score (mAP50 0.99) comes
  from a tiny validation set and is not a reliable accuracy figure.
- It was tested on the held-out part of one ambulance/fire-engine clip and on part of
  `cctv.mp4`. It has not been tested on a completely new clip.

## Vehicle type classifier

- Trained on a small set of crops sorted by hand. Truck (6 crops) and taxi (5 crops)
  are weak classes.

## Violations

- Only wrong way and stopped vehicle are checked. `cctv.mp4` contains no real violation,
  so these rules were checked only with made-up data and with one false alarm that was
  fixed (a taxi waiting at the kerb; the left 30% of the frame is ignored).
