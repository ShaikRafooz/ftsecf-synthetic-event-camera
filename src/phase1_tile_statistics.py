"""FTSECF Phase 1: tile statistics engine.

Captures one frame, splits it into 32x32 tiles, and computes the mean, standard
deviation and a preliminary statistical threshold (mean + K * std) for each tile.
Saves a 3-panel figure to results/phase1_tile_statistics.png.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np

from common import RESULTS_DIR, FrameSource, add_source_args, to_gray_cropped

TILE_SIZE = 32
K = 2


def compute_tile_maps(gray):
    """Return (mean_map, std_map, threshold_map), one value per tile."""
    tile_rows = gray.shape[0] // TILE_SIZE
    tile_cols = gray.shape[1] // TILE_SIZE
    mean_map = np.zeros((tile_rows, tile_cols), dtype=np.float32)
    std_map = np.zeros((tile_rows, tile_cols), dtype=np.float32)
    threshold_map = np.zeros((tile_rows, tile_cols), dtype=np.float32)

    for i in range(tile_rows):
        for j in range(tile_cols):
            tile = gray[i * TILE_SIZE:(i + 1) * TILE_SIZE,
                        j * TILE_SIZE:(j + 1) * TILE_SIZE]
            mu = np.mean(tile)
            sigma = np.std(tile)
            mean_map[i, j] = mu
            std_map[i, j] = sigma
            threshold_map[i, j] = mu + K * sigma
    return mean_map, std_map, threshold_map


def main():
    parser = add_source_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    if args.no_display:
        plt.use("Agg")
    import matplotlib.pyplot as plt

    source = FrameSource(video=args.video, camera=args.camera)
    frame = source.read()
    source.release()
    if frame is None:
        raise SystemExit("Could not read a frame.")

    gray = to_gray_cropped(frame, TILE_SIZE)
    tile_rows, tile_cols = gray.shape[0] // TILE_SIZE, gray.shape[1] // TILE_SIZE
    print(f"Frame: {gray.shape} | Tile grid: {tile_rows}x{tile_cols} = {tile_rows * tile_cols} tiles")

    mean_map, std_map, threshold_map = compute_tile_maps(gray)
    print(f"Threshold map - Min: {threshold_map.min():.1f}, Max: {threshold_map.max():.1f}")
    print(f"Mean map - Min: {mean_map.min():.1f}, Max: {mean_map.max():.1f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("FTSECF - Phase 1: Tile Statistics Engine", fontsize=14)
    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("Original Frame (Grayscale)")
    axes[0].axis("off")
    im1 = axes[1].imshow(mean_map, cmap="hot", interpolation="nearest")
    axes[1].set_title("Mean Intensity per Tile")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1])
    im2 = axes[2].imshow(threshold_map, cmap="hot", interpolation="nearest")
    axes[2].set_title(f"Threshold Map (mean + {K} x std)")
    axes[2].axis("off")
    plt.colorbar(im2, ax=axes[2])
    plt.tight_layout()

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "phase1_tile_statistics.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    if not args.no_display:
        plt.show()


if __name__ == "__main__":
    main()
