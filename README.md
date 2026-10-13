# FTSECF(Frame Threshold Synthesis Event Capture Fusion): Synthetic Event Camera from a Webcam

Simulates an event camera (ON/OFF events from frame-to-frame brightness change) using a
standard webcam or a video file, and compares a fixed threshold with a per-tile adaptive
threshold based on a noise model. Python, OpenCV, NumPy.

[Phase 4 sample](docs/results/frame_0080_t15.0s_chg23.28pct.png)


## How it works

```
D = P_t - P_(t-1)                      (int32, so negative changes are kept)
ON if D > T,  OFF if D < -T
T[i,j] = k * sqrt(sigma_read^2 + alpha * mu[i,j])
         k = 3, sigma_read = 1.0, alpha = 0.5, mu = mean of a 32x32 tile (15x20 tiles)
```

Dark tiles get a lower threshold and bright tiles a higher one, because sensor noise grows
with brightness.

| Phase | File | What it does |
|---|---|---|
| 1 | `src/phase1_tile_statistics.py` | Per-tile mean, std and a statistical threshold map |
| 2 | `src/phase2_dual_pipeline.py` | Fixed vs. adaptive (mean + 2 std) events side by side |
| 3 | `src/phase3_physics_threshold.py` | Noise-model threshold, loop-based |
| 4 | `src/phase4_recorder.py` | Vectorized version; saves a frame only if at least 5% of pixels fire |

Phase 2's adaptive threshold compares an intensity statistic with a frame *difference*, so it
almost never fires. Phase 3 replaces it with a noise-based threshold.

## Run

To run the programs follow the steps below:
```
pip install -r requirements.txt
cd src
python make_test_video.py                                # creates sample.avi, no webcam needed
python phase4_recorder.py --video sample.avi --no-display
python phase4_recorder.py --camera 0                     # live webcam (Q quits the preview)
```

Options shared by all phases: `--video FILE`, `--camera N`, `--no-display`, `--max-frames N`.
Phase 4 also takes `--seconds` and `--out-dir`. Results of phases 1-3 are saved in `results/`.

## Results (my laptop, 640x480 webcam)

- Processing: about 18-23 ms per frame (roughly 45-55 FPS of capacity). End to end: about
  22-26 FPS with frame saving, below the camera's 30 FPS.
- Phase 4 discards frames when little changes. The share depends on the scene:

| Scene | Frames discarded |
|---|---|
| Camera panned around a room | 53.2% |
| Still wall with a hand moving in and out | 57.6% |
| Low-motion scene | 90.3% |

- On a still wall, both pipelines fire almost no events (0-4 per frame).

## Limitations

- alpha = 0.5 was set empirically, not measured. My webcam showed fewer false events than
  the model predicts.
- The "3 sigma = 0.3% false alarms" figure holds for a single frame. A difference of two
  frames has twice the variance, so the real rate is about 3.4%.
- Linear intensity is used; real event cameras respond to log intensity.
- Event counts compare the pipelines but are not ground-truth accuracy.
- No controlled noise-reduction test yet. Planned: a still scene to count false events, and
  a known weak target to count detections.

## Team and credits

Theme-based project at Vasavi College of Engineering (2025-26), advised by
Dr. P Sathish and Dr. K Deepti.

| Member | Role |
|---|---|
| Ibaad Ghori | Phase 1 (tile statistics) and project documentation |
| Velide Shanmukha Srinivas | Phase 2 (dual-pipeline event simulator) and simulation of all phases |
| Shaik Rafooz | Phase 3 (noise-model threshold) and Phase 4 (vectorized recorder) |

The code in this repo is a cleaned-up version of the project's original scripts,
refactored and packaged by Shaik Rafooz (video input, shared helpers, bug fixes), who also
ran the measurements reported above.
