"""Shared helpers for the FTSECF scripts: frame source (webcam or video file) and tile cropping."""
import sys
from pathlib import Path

import cv2

WIDTH, HEIGHT = 640, 480
REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"


def add_source_args(parser):
    """Add the command-line options every phase script shares."""
    parser.add_argument("--video", default=None,
                        help="video file to process instead of a webcam")
    parser.add_argument("--camera", type=int, default=1,
                        help="webcam index (default: 1)")
    parser.add_argument("--no-display", action="store_true",
                        help="headless mode: do not open any windows")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="stop after this many frames")
    return parser


class FrameSource:
    """Reads BGR frames at 640x480 from a webcam or a video file."""

    def __init__(self, video=None, camera=0, warmup=30):
        self.is_video = video is not None
        if self.is_video:
            self.cap = cv2.VideoCapture(video)
        else:
            # DirectShow gives low-latency capture on Windows; other OSes use the default backend.
            backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
            self.cap = cv2.VideoCapture(camera, backend)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open {'video file ' + video if self.is_video else 'camera'}.")

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and fps > 0 else 30.0

        if not self.is_video:
            print("Camera warming up...")  # lets auto-exposure settle
            for _ in range(warmup):
                self.cap.read()
            print("Done.")

    def read(self):
        """Return the next frame, or None when the source ends or fails."""
        ok, frame = self.cap.read()
        if not ok:
            return None
        if frame.shape[1] != WIDTH or frame.shape[0] != HEIGHT:
            frame = cv2.resize(frame, (WIDTH, HEIGHT))
        return frame

    def release(self):
        self.cap.release()


def to_gray_cropped(frame, tile_size):
    """Convert to grayscale and crop to an exact multiple of the tile size."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    return gray[: h - h % tile_size, : w - w % tile_size]
