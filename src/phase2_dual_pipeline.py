"""FTSECF Phase 2: dual-pipeline event simulator.

Computes the signed difference between consecutive frames and generates ON/OFF
events two ways, side by side:
  - fixed pipeline:    one global threshold for every pixel
  - adaptive pipeline: per-tile threshold = mean + K * std (Phase 1 statistics)
Press Q in the window to quit. The last comparison image is saved to
results/phase2_event_comparison.png.
"""
import argparse

import cv2
import numpy as np

from common import RESULTS_DIR, FrameSource, add_source_args, to_gray_cropped

TILE_SIZE = 32
K = 2
FIXED_THRESH = 30


def compute_threshold_map(gray_frame):
    """Per-tile threshold (mean + K * std), expanded to full-frame size."""
    h, w = gray_frame.shape
    t_rows, t_cols = h // TILE_SIZE, w // TILE_SIZE
    T = np.zeros((h, w), dtype=np.float32)
    for i in range(t_rows):
        for j in range(t_cols):
            tile = gray_frame[i * TILE_SIZE:(i + 1) * TILE_SIZE,
                              j * TILE_SIZE:(j + 1) * TILE_SIZE]
            T[i * TILE_SIZE:(i + 1) * TILE_SIZE,
              j * TILE_SIZE:(j + 1) * TILE_SIZE] = np.mean(tile) + K * np.std(tile)
    return T


def generate_events(diff, threshold_map):
    """ON events (brightness up) in blue, OFF events (brightness down) in red."""
    event_map = np.zeros((diff.shape[0], diff.shape[1], 3), dtype=np.uint8)
    on_mask = diff > threshold_map
    off_mask = diff < -threshold_map
    event_map[on_mask] = [255, 0, 0]   # BGR: blue
    event_map[off_mask] = [0, 0, 255]  # BGR: red
    return event_map, int(np.sum(on_mask)), int(np.sum(off_mask))


def main():
    parser = add_source_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()

    source = FrameSource(video=args.video, camera=args.camera)
    frame = source.read()
    if frame is None:
        raise SystemExit("Could not read a frame.")
    # int32 is essential: uint8 subtraction would wrap around.
    frame_prev = to_gray_cropped(frame, TILE_SIZE).astype(np.int32)
    print("Running FTSECF dual-pipeline loop" + ("" if args.no_display else " - press Q to quit"))

    combined = None
    frame_count = 0
    while args.max_frames is None or frame_count < args.max_frames:
        frame = source.read()
        if frame is None:
            break
        gray = to_gray_cropped(frame, TILE_SIZE)
        frame_t = gray.astype(np.int32)
        diff = frame_t - frame_prev

        fixed_map = np.full(diff.shape, FIXED_THRESH, dtype=np.float32)
        events_fixed, on_f, off_f = generate_events(diff, fixed_map)
        events_adaptive, on_a, off_a = generate_events(diff, compute_threshold_map(gray))

        frame_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        overlay_fixed = cv2.addWeighted(frame_bgr, 0.5, events_fixed, 0.5, 0)
        overlay_adaptive = cv2.addWeighted(frame_bgr, 0.5, events_adaptive, 0.5, 0)
        cv2.putText(overlay_fixed, f"FIXED thresh={FIXED_THRESH} | ON:{on_f} OFF:{off_f}",
                    (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(overlay_adaptive, f"ADAPTIVE k={K} | ON:{on_a} OFF:{off_a}",
                    (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        combined = np.hstack([overlay_fixed, overlay_adaptive])

        if frame_count % 10 == 0:
            print(f"Fixed - ON:{on_f:5d} OFF:{off_f:5d} | Adaptive - ON:{on_a:5d} OFF:{off_a:5d}")

        if not args.no_display:
            cv2.imshow("LEFT: Fixed Threshold | RIGHT: Adaptive (FTSECF)", combined)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        frame_prev = frame_t
        frame_count += 1

    source.release()
    if not args.no_display:
        cv2.destroyAllWindows()
    if combined is not None:
        RESULTS_DIR.mkdir(exist_ok=True)
        out_path = RESULTS_DIR / "phase2_event_comparison.png"
        cv2.imwrite(str(out_path), combined)
        print(f"Saved last frame: {out_path}")
    print(f"Done. Processed {frame_count} frames.")


if __name__ == "__main__":
    main()
