"""FTSECF Phase 4: NumPy-vectorized pipeline with automatic change detection.

Runs the physics-based adaptive threshold (vectorized, no Python loops) and saves a
frame only if at least CHANGE_THRESHOLD of its pixels fire an adaptive event.
Saved frames are annotated with the regime (SENSITIVITY GAIN / NOISE REDUCTION),
event counts for both pipelines, timestamp, FPS and change percentage.
With a video file, timestamps follow the video's own clock.
"""
import argparse
import os
import time

import cv2
import numpy as np

from common import FrameSource, add_source_args, to_gray_cropped

TILE_SIZE = 32
K = 3
FIXED_THRESH = 30
READ_NOISE = 1.0
SHOT_COEFF = 0.5
TILE_ROWS = 480 // TILE_SIZE   # 15
TILE_COLS = 640 // TILE_SIZE   # 20
CAPTURE_SECONDS = 20           # default recording duration
CHANGE_THRESHOLD = 0.05        # 5% of pixels must fire to keep a frame (15,360 px at 640x480)
TOTAL_PIXELS = TILE_ROWS * TILE_SIZE * TILE_COLS * TILE_SIZE


def compute_threshold_map_vectorized(gray_frame):
    """Same model as Phase 3, but computed for all tiles at once."""
    tiles = gray_frame.reshape(TILE_ROWS, TILE_SIZE, TILE_COLS, TILE_SIZE)
    tile_means = tiles.mean(axis=(1, 3))
    T_tiles = K * np.sqrt(READ_NOISE ** 2 + SHOT_COEFF * tile_means)
    return np.repeat(np.repeat(T_tiles, TILE_SIZE, axis=0), TILE_SIZE, axis=1).astype(np.float32)


def generate_events(diff, threshold_map):
    event_map = np.zeros((*diff.shape, 3), dtype=np.uint8)
    on_mask = diff > threshold_map
    off_mask = diff < -threshold_map
    event_map[on_mask] = [255, 0, 0]   # BGR: blue = ON
    event_map[off_mask] = [0, 0, 255]  # BGR: red = OFF
    return event_map, int(np.sum(on_mask)), int(np.sum(off_mask))


def main():
    parser = add_source_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--seconds", type=float, default=CAPTURE_SECONDS, help="recording duration")
    parser.add_argument("--out-dir", default="ftsecf_captured_frames", help="folder for saved frames")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Frames with change will be saved to: {args.out_dir}/")
    source = FrameSource(video=args.video, camera=args.camera)
    print(f"Done. Recording for {args.seconds:g} seconds...")
    print(f"Change threshold: {CHANGE_THRESHOLD * 100:.1f}% of pixels")
    print("-" * 42)

    fixed_map = np.full((TILE_ROWS * TILE_SIZE, TILE_COLS * TILE_SIZE), FIXED_THRESH, dtype=np.float32)
    frame = source.read()
    if frame is None:
        raise SystemExit("Could not read a frame.")
    frame_prev = to_gray_cropped(frame, TILE_SIZE).astype(np.int32)

    frame_count = saved_count = 0
    proc_times = []   # seconds of pure computation per frame (excludes camera wait)
    start_time = time.time()

    while args.max_frames is None or frame_count < args.max_frames:
        # Video files use the video's clock; a webcam uses wall-clock time.
        elapsed = frame_count / source.fps if source.is_video else time.time() - start_time
        if elapsed >= args.seconds:
            print(f"\nCapture complete - {args.seconds:g}s elapsed.")
            break

        t_start = time.time()
        frame = source.read()
        if frame is None:
            print("\nNo more frames (end of video or lost camera).")
            break

        t_proc = time.perf_counter()
        gray = to_gray_cropped(frame, TILE_SIZE)
        frame_t = gray.astype(np.int32)
        diff = frame_t - frame_prev

        ev_fixed, on_f, off_f = generate_events(diff, fixed_map)
        ev_adapt, on_a, off_a = generate_events(diff, compute_threshold_map_vectorized(gray))
        total_fixed, total_adapt = on_f + off_f, on_a + off_a
        change_fraction = total_adapt / TOTAL_PIXELS
        proc_times.append(time.perf_counter() - t_proc)

        frame_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        overlay_adapt = cv2.addWeighted(frame_bgr, 0.5, ev_adapt, 0.5, 0)

        if change_fraction >= CHANGE_THRESHOLD:
            overlay_fixed = cv2.addWeighted(frame_bgr, 0.5, ev_fixed, 0.5, 0)
            if total_fixed > total_adapt:
                pct = (total_fixed - total_adapt) / max(total_fixed, 1) * 100
                label, color = f"NOISE REDUCTION: {pct:.1f}%", (0, 255, 0)
            else:
                pct = (total_adapt - total_fixed) / max(total_adapt, 1) * 100
                label, color = f"SENSITIVITY GAIN: {pct:.1f}%", (0, 255, 255)

            cv2.putText(overlay_fixed, f"FIXED={FIXED_THRESH} | Events:{total_fixed}",
                        (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            cv2.putText(overlay_adapt, f"ADAPTIVE k={K} | Events:{total_adapt}",
                        (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            combined = np.hstack([overlay_fixed, overlay_adapt])
            cv2.putText(combined, label, (10, combined.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(combined,
                        f"t={elapsed:.1f}s | Change:{change_fraction * 100:.2f}% | Frame#{frame_count}",
                        (10, combined.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            filename = os.path.join(
                args.out_dir,
                f"frame_{saved_count:04d}_t{elapsed:.1f}s_chg{change_fraction * 100:.2f}pct.png")
            cv2.imwrite(filename, combined)
            saved_count += 1
            print(f"  [t={elapsed:5.1f}s | f#{frame_count:4d}] SAVED #{saved_count:3d} | "
                  f"Fixed:{total_fixed:5d} | Adaptive:{total_adapt:5d} | "
                  f"Change:{change_fraction * 100:.2f}% | {label}")
        else:
            print(f"  [t={elapsed:5.1f}s | f#{frame_count:4d}] static "
                  f"(change={change_fraction * 100:.3f}% < {CHANGE_THRESHOLD * 100:.1f}%)", end="\r")

        if not args.no_display:
            time_left = args.seconds - elapsed
            cv2.putText(overlay_adapt, f"Recording... {time_left:.0f}s left | Saved: {saved_count}",
                        (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(overlay_adapt,
                        f"Change: {change_fraction * 100:.3f}% (threshold: {CHANGE_THRESHOLD * 100:.1f}%)",
                        (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
            cv2.imshow("FTSECF Phase 4 - Live Preview (Q to stop early)", overlay_adapt)

        if not args.no_display and cv2.waitKey(1) & 0xFF == ord("q"):
            print("\nStopped early by user.")
            break
        frame_prev = frame_t
        frame_count += 1

    source.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    print("\n-- SESSION SUMMARY " + "-" * 30)
    print(f"  Total frames     : {frame_count}")
    print(f"  Frames saved     : {saved_count}")
    print(f"  Frames discarded : {frame_count - saved_count}")
    if frame_count > 0:
        kept_pct = saved_count / frame_count * 100
        print(f"  Discarded share  : {100 - kept_pct:.1f}%")
        print(f"  Kept share       : {kept_pct:.1f}%")
    total_time = time.time() - start_time
    if frame_count > 0 and total_time > 0:
        print(f"  Overall FPS      : {frame_count / total_time:.1f}  (frames / total time)")
    if proc_times:
        mean_proc = np.mean(proc_times)
        print(f"  Processing time  : {mean_proc * 1000:.1f} ms per frame "
              f"(capacity ~{1 / mean_proc:.0f} FPS)")
    print(f"  Output folder    : {args.out_dir}/")


if __name__ == "__main__":
    main()