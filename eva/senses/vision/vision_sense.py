from config import logger
import asyncio
from pathlib import Path
from typing import Dict, List
import numpy as np
import cv2

from eva.senses.vision.describer import Describer
from eva.senses.vision.identifier import Identifier
from eva.senses.vision.webcam import Webcam
from eva.senses.sense_buffer import SenseBuffer


class CameraSense:
    """Background vision — EVA's eyes.

    Continuously captures frames, detects scene changes,
    and writes descriptions to a shared input buffer.
    """

    _PIXEL_DIFF_THRESHOLD = 10 # The threshold for detecting significant pixel differences.
    _CHANGE_THRESHOLD = 0.4 # The threshold for detecting significant scene changes.
    _GLANCE_INTERVAL = 5.0 # The interval between camera glances, adjusted to balance performance.
    _COMPRESSED_SIZE = (320, 240) # The size of the image to be processed.
    
    def __init__(
        self,
        describer: Describer,
        identifier: Identifier | None = None,
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

        self._previous_frame: np.ndarray | None = None
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task | None = None

    @property
    def is_available(self) -> bool:
        return self.webcam.camera is not None

    def start(self, buffer: SenseBuffer) -> None:
        """Start the background vision task, writing observations to buffer."""
        if self.webcam.camera is None:
            logger.warning("CameraSense: No camera — skipping vision.")
            return
        if self._task is not None and not self._task.done():
            logger.warning("CameraSense: Already running.")
            return

        self._stop_event.clear()
        self._task = asyncio.create_task(self._run(buffer))
        logger.debug("CameraSense: Started.")

    async def stop(self) -> None:
        """Stop the vision task and release resources."""
        if self._task is None:
            self.webcam.release()
            return

        if self._task.done():
            self._task = None
            self.webcam.release()
            return

        logger.debug("CameraSense: Stopping...")
        self._stop_event.set()
        try:
            await self._task
        finally:
            self._task = None
            self.webcam.release()
        logger.debug("CameraSense: Stopped.")

    async def _run(self, buffer: SenseBuffer) -> None:
        """Main loop: capture -> detect change -> describe -> write to buffer."""

        while not self._stop_event.is_set():
            try:
                frame = await asyncio.to_thread(self.webcam.capture_photo)

                if self._has_scene_changed(frame):
                    logger.debug("CameraSense: Scene change detected, observing...")
                    observation, face_ids = await self._observe(frame)

                    if observation:
                        metadata = {"faces": face_ids} if face_ids else None
                        buffer.push("observation", observation, metadata=metadata)
                        logger.debug(f"CameraSense: {observation}...")

            except Exception as e:
                logger.error(f"CameraSense: Capture loop error — {e}")

            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self._GLANCE_INTERVAL)
            except asyncio.TimeoutError:
                pass

        logger.debug("CameraSense: Capture loop stopped.")

    def _has_scene_changed(self, frame: np.ndarray) -> bool:
        """Motion detection via grayscale pixel diff."""
        # Resize first for efficiency and noise reduction
        small = cv2.resize(frame, self._COMPRESSED_SIZE)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        if self._previous_frame is None:
            self._previous_frame = gray
            return True

        diff = cv2.absdiff(gray, self._previous_frame)
        changed_pixels = np.count_nonzero(diff > self._PIXEL_DIFF_THRESHOLD)
        change_ratio = changed_pixels / diff.size
        self._previous_frame = gray

        return change_ratio > self._CHANGE_THRESHOLD

    async def _observe(self, frame: np.ndarray) -> tuple[str | None, List[str]]:
        """Run description and face identification in parallel, combine results.

        Returns (observation_text, face_ids) — face_ids are PeopleDB IDs for recognized faces.
        """

        description, faces = await asyncio.gather(
            self._describe(frame),
            self._identify(frame),
            return_exceptions=True,
        )
        
        description = description if not isinstance(description, BaseException) else None
        faces = faces if not isinstance(faces, BaseException) else []

        if not description and not faces:
            return None, []

        # Extract face IDs for known people
        face_ids = [f["id"] for f in faces if f.get("id")]

        # Combine: "[Alice, unknown person] Two people at desk, one waving"
        parts = []
        if faces:
            names = [f["name"] for f in faces]
            parts.append(f"[{', '.join(names)}]")
        if description:
            parts.append(description)

        return (" ".join(parts) if parts else None), face_ids

    async def _describe(self, frame: np.ndarray) -> str | None:
        """Describe frame using vision model (async network I/O)."""
        resized = cv2.resize(frame, self._COMPRESSED_SIZE)
        return await self.describer.describe(resized)

    async def _identify(self, frame: np.ndarray) -> List[Dict]:
        """Identify faces in frame (CPU-bound via thread pool)."""
        if self.identifier is None:
            return []
        return await asyncio.to_thread(self.identifier.identify, frame)

    def capture(self, save_path: str) -> None:
        """On-demand photo capture (for face registration etc)."""
        try:
            frame = self.webcam.capture_photo()
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), frame)
            logger.debug(f"CameraSense: Photo saved to {path}")
        except Exception as e:
            logger.error(f"CameraSense: Photo capture error — {e}")
