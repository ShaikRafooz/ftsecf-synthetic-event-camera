"""FTSECF Phase 3: physics-based adaptive threshold.

Replaces the statistical threshold with a photon-transfer noise model:
    T[i,j] = K * sqrt(READ_NOISE^2 + SHOT_COEFF * mean[i,j])
Point a webcam at a bright lamp for the noise-reduction regime; keep it in a dim
room for the sensitivity-gain regime. A [DIAG] line is printed every 30 frames.
"""
import argparse

import cv2
import numpy as np

from common import RESULTS_DIR, FrameSource, add_source_args, to_gray_cropped

TILE_SIZE = 32
K = 3               # 3-sigma detection
FIXED_THRESH = 15   # global threshold for the comparison pipeline
READ_NOISE = 1.0    # counts, from a dark-frame measurement
SHOT_COEFF = 0.5    # empirically calibrated for the Lenovo 300 FHD webcam


def compute_threshold_map_physics(gray_frame):
    """Per-tile threshold T[i,j] = K * sqrt(READ_NOISE^2 + SHOT_COEFF * mean[i,j])."""
    h, w = gray_frame.shape
    t_rows, t_cols = h // TILE_SIZE, w // TILE_SIZE
    T = np.zeros((h, w), dtype=np.float32)
    for i in range(t_rows):
        for j in range(t_cols):
            tile = gray_frame[i * TILE_SIZE:(i + 1) * TILE_SIZE,
                              j * TILE_SIZE:(j + 1) * TILE_SIZE]
            mu = np.mean(tile)
            noise = np.sqrt(READ_NOISE ** 2 + SHOT_COEFF * mu)
            T[i * TILE_SIZE:(i + 1) * TILE_SIZE,
              j * TILE_SIZE:(j + 1) * TILE_SIZE] = K * noise
    return T


def generate_events(diff, threshold_map):
    event_map = np.zeros((*diff.shape, 3), dtype=np.uint8)
    on_mask = diff > threshold_map
    off_mask = diff < -threshold_map
    event_map[on_mask] = [255, 0, 0]   # BGR: blue = ON
    event_map[off_mask] = [0, 0, 255]  # BGR: red = OFF
    return event_map, int(np.sum(on_mask)), int(np.sum(off_mask))


def summary(total_f, total_a):
    if total_f == 0 and total_a == 0:
        return "Static scene - zero events in both pipelines"
    if total_f > total_a:
        pct = (total_f - total_a) / max(total_f, 1) * 100
        return f"Fixed:{total_f:6d} | Adaptive:{total_a:6d} | Noise reduction: {pct:.1f}%"
    pct = (total_a - total_f) / max(total_a, 1) * 100
    return f"Fixed:{total_f:6d} | Adaptive:{total_a:6d} | Sensitivity gain: {pct:.1f}%"


def main():
    parser = add_source_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()

    source = FrameSource(video=args.video, camera=args.camera)
    frame = source.read()
    if frame is None:
        raise SystemExit("Could not read a frame.")
    gray = to_gray_cropped(frame, TILE_SIZE)
    frame_prev = gray.astype(np.int32)
    fixed_map = np.full(frame_prev.shape, FIXED_THRESH, dtype=np.float32)
    print("Running Phase 3 - physics-based noise model" + ("" if args.no_display else " - press Q to quit"))

    combined = None
    frame_count = 0
    while args.max_frames is None or frame_count < args.max_frames:
        frame = source.read()
        if frame is None:
            break
        gray = to_gray_cropped(frame, TILE_SIZE)
        frame_t = gray.astype(np.int32)
        diff = frame_t - frame_prev

        ev_fixed, on_f, off_f = generate_events(diff, fixed_map)
        T_map = compute_threshold_map_physics(gray)
        ev_adapt, on_a, off_a = generate_events(diff, T_map)

        if frame_count % 30 == 0:
            t_min, t_max = float(T_map.min()), float(T_map.max())
            status = (">>> CROSSOVER ACTIVE - noise reduction possible <<<" if t_max > FIXED_THRESH
                      else f"need brighter tiles (T_max={t_max:.1f} < FIXED={FIXED_THRESH})")
            print(f"  [DIAG frame {frame_count}] pixel max={int(gray.max())} "
                  f"mean={np.mean(gray):.1f} | T_adaptive: {t_min:.1f}-{t_max:.1f} | {status}")
        if frame_count % 10 == 0:
            print(summary(on_f + off_f, on_a + off_a))

        frame_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        overlay_fixed = cv2.addWeighted(frame_bgr, 0.5, ev_fixed, 0.5, 0)
        overlay_adapt = cv2.addWeighted(frame_bgr, 0.5, ev_adapt, 0.5, 0)
        for img, name, on, off in ((overlay_fixed, f"FIXED={FIXED_THRESH}", on_f, off_f),
                                   (overlay_adapt, f"ADAPTIVE k={K}", on_a, off_a)):
            cv2.putText(img, name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(img, f"ON:{on}", (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 100, 0), 1, cv2.LINE_AA)
            cv2.putText(img, f"OFF:{off}", (8, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA)
        combined = np.hstack([overlay_fixed, overlay_adapt])

        if not args.no_display:
            cv2.imshow("LEFT: Fixed | RIGHT: Physics FTSECF", combined)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        frame_prev = frame_t
        frame_count += 1

    source.release()
    if not args.no_display:
        cv2.destroyAllWindows()
    if combined is not None:
        RESULTS_DIR.mkdir(exist_ok=True)
        out_path = RESULTS_DIR / "phase3_physics_comparison.png"
        cv2.imwrite(str(out_path), combined)
        print(f"Saved last frame: {out_path}")
    print(f"Phase 3 complete. Processed {frame_count} frames.")


if __name__ == "__main__":
    main()
