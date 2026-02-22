"""Simplified video input (adapted from Autobahn)."""
from __future__ import annotations

import logging
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

logger = logging.getLogger(__name__)


@dataclass
class VideoMeta:
    fps: float
    width: int
    height: int
    frame_count: int


@dataclass
class FramePacket:
    frame: np.ndarray
    timestamp: float


class VideoInput:
    """Simple video file reader for post-hoc accident analysis."""

    def __init__(self, path: str | Path, frame_rate: int | None = None):
        self.path = Path(path)
        self.frame_rate = frame_rate
        self.cap = None
        self.meta: VideoMeta | None = None

        if cv2 is None:
            raise ImportError("opencv-python is required for VideoInput")
        if not self.path.exists():
            raise FileNotFoundError(f"Video not found: {self.path}")

        self.cap = cv2.VideoCapture(str(self.path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {self.path}")

        self.meta = VideoMeta(
            fps=float(self.cap.get(cv2.CAP_PROP_FPS) or (frame_rate or 30.0)),
            width=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            frame_count=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        )
        logger.info("Video opened: %s fps=%.2f size=%dx%d frames=%d", self.path, self.meta.fps, self.meta.width, self.meta.height, self.meta.frame_count)

    def frames(self) -> Generator[tuple[int, FramePacket], None, None]:
        if self.cap is None:
            return
        idx = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            idx += 1
            yield idx, FramePacket(frame=frame, timestamp=idx / (self.meta.fps if self.meta else 30.0))

    def release(self) -> None:
        if self.cap:
            self.cap.release()
            logger.info("Closed video %s", self.path)
