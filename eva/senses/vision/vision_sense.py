from config import logger
import asyncio
from pathlib import Path
from typing import Dict, List
import numpy as np
import cv2

from eva.senses.vision.describer import Describer
from eva.senses.vision.face_identifier import FaceIdentifier
from eva.senses.vision.webcam import Webcam


class CameraSense:
    """EVA's eyes — a passive vision faculty: capture, describe, identify on demand."""

    def __init__(
        self,
        describer: Describer,
        identifier: FaceIdentifier | None = None,
        source: int | str = 0,
    ):
        """
        Args:
            source: int for local V4L2 device, or URL string for network
                    stream (e.g. "http://198.18.0.1:5000/video" for WSL).
        """
        self.describer = describer
        self.identifier = identifier
        self.webcam = Webcam(source)

    @property
    def is_available(self) -> bool:
        return self.webcam.camera is not None

    def capture_photo(self) -> np.ndarray:
        """Grab a single frame. Sync/blocking cv2"""
        return self.webcam.capture_photo()

    async def describe(self, frame: np.ndarray, names: list[str] | None = None) -> str | None:
        """Describe a frame with the vision model. """
        return await self.describer.describe(frame, names=names)

    async def identify(self, frame: np.ndarray) -> List[Dict]:
        """Identify faces in a frame. Empty list when no identifier."""
        if self.identifier is None:
            return []
        return await asyncio.to_thread(self.identifier.identify, frame)

    def release(self) -> None:
        """Release the webcam and vision models."""
        self.webcam.release()
        if self.identifier:
            self.identifier.close()

    def capture(self, save_path: str) -> None:
        """On-demand photo capture to disk (for face registration etc)."""
        try:
            frame = self.webcam.capture_photo()
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), frame)
            logger.debug(f"CameraSense: Photo saved to {path}")
        except Exception as e:
            logger.error(f"CameraSense: Photo capture error — {e}")
