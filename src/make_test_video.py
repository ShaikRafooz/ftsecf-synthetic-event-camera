"""Generate a synthetic test video so the FTSECF scripts can run without a webcam.

The scene is a horizontal brightness ramp (dim on the left, bright on the right) with
signal-dependent sensor noise, plus a textured patch that moves between 3-8 s and
13-18 s. The rest of the time the scene is static. This is NOT real sensor data; it is
only meant to check that the pipeline runs and behaves sensibly.
"""
import argparse

import cv2
import numpy as np

W, H, FPS = 640, 480, 30
READ_NOISE, SHOT_COEFF = 1.0, 0.5  # same noise model the thresholds assume


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="sample.avi")
    p.add_argument("--seconds", type=int, default=20)
    p.add_argument("--noise-scale", type=float, default=1.0,
                   help="multiply the noise level (1.0 = model noise)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    ramp = np.tile(np.linspace(15, 235, W, dtype=np.float32), (H, 1))
    texture = cv2.GaussianBlur(rng.uniform(40, 240, (180, 240)).astype(np.float32), (0, 0), 1)

    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    for f in range(args.seconds * FPS):
        t = f / FPS
        scene = ramp.copy()
        if 3 <= t < 8 or 13 <= t < 18:
            x = int(200 + 150 * np.sin(2 * np.pi * t / 2.5))
            scene[150:330, x:x + 240] = texture
        sigma = np.sqrt(READ_NOISE ** 2 + SHOT_COEFF * scene) * args.noise_scale
        noisy = np.clip(scene + rng.normal(0, 1, scene.shape) * sigma, 0, 255).astype(np.uint8)
        writer.write(cv2.cvtColor(noisy, cv2.COLOR_GRAY2BGR))
    writer.release()
    print(f"Wrote {args.out} ({args.seconds}s, {W}x{H} @ {FPS} FPS)")


if __name__ == "__main__":
    main()
