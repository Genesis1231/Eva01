"""EVA's subconscious: the always-on perceptual gate beneath the conscious graph.

Raw frames never enter the SenseBuffer, they stay inside the gate, so they don't reset the Heart's
idle clock. Only a salient scene reaches the brain.
"""

import asyncio
import time
from pathlib import Path

import cv2

from config import logger
from eva.utils.format import image_uri
from eva.senses.sense_buffer import SenseBuffer
from eva.senses.vision.vision_sense import CameraSense
from eva.subconscious._vision.detector import CamEvent, VisionDetector


class Subconscious:
    """The always-on perceptual gate: an async ~1 fps loop (a peer of breathe in wake(), not a
    graph node). Each glance routes: inspect wakes the brain, acknowledge habituates quietly."""

    GATE_INTERVAL = 1.0     # seconds per glance (~1 fps)
    MAX_BACKOFF = 60.0      # glance interval cap while the camera/engine is down
    def __init__(
        self,
        sense_buffer: SenseBuffer,
        vision_sense: CameraSense,
        vision_detector: VisionDetector,
        inspect_dir: Path,
        interval: float = GATE_INTERVAL,
    ):
        self.sense_buffer = sense_buffer
        self.vision = vision_sense
        self.vision_detector = vision_detector
        self.inspect_dir = inspect_dir
        self.interval = interval
        self._stop = asyncio.Event()
        
        logger.debug("Subconscious: initialized.")

    async def start(self) -> None:
        """Beat forever: glance, gate, surface. A peer in wake()'s asyncio.gather."""
        
        if not self.vision.is_available:
            logger.warning("Subconscious: no camera, vision gate disabled.")
            return

        logger.debug(f"Subconscious started. vision gate at ~{1 / self.interval:.1f} fps")
        failures = 0
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                healthy = await self._glance()
                failures = 0 if healthy else failures + 1
            except Exception:
                failures += 1
                logger.error("Subconscious: gate error", exc_info=True)

            # A dead camera or embedding engine fails every glance; back off
            # exponentially instead of spinning a hot error loop, recover on success.
            delay = min(self.interval * 2 ** failures, self.MAX_BACKOFF)
            await self._pace(delay - (time.monotonic() - started))

    async def _pace(self, remaining: float) -> None:
        """Sleep the rest of the interval, but wake immediately on stop, holding ~1 fps."""
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=max(0.0, remaining))
        except asyncio.TimeoutError:
            pass
        
    async def _glance(self) -> bool:
        """Inspect wakes the brain; acknowledge habituates. Returns False when
        perception failed (engine down) so the gate loop can back off."""

        frame = await asyncio.to_thread(self.vision.capture_photo)
        view = await self.vision_detector.observe(frame, time.monotonic())

        if view is None:
            return False
        if view.event is None:
            return True

        if view.event.level == 2:
            # inspect: genuinely new, wake the brain
            await self._surface(view.event)
        else:
            # acknowledge: familiar, habituate quietly
            logger.debug(f"Subconscious: familiar habituation: novelty_z={view.event.novelty_z:.2f}, long_nov={view.event.long_nov:.3f}")
        return True



    async def _surface(self, event: CamEvent) -> None:
        """Wake the brain on a novel scene: save the frame, then push the raw frame itself (as a
        data-URI) to the SenseBuffer so the cortex sees the image directly (cognitive mode), with no
        Describer caption in the gate path."""
        
        self._save_inspect(event) # keep the novel frame
        
        # description = await self.vision.describe(event.frame)
        # if not description:
        #     logger.warning("Describer: Failed to describe a novel scene.")
        #     description = "I saw something new, but I couldn't describe it."
        
        _, buffer = cv2.imencode('.jpg', event.frame)
        frame_uri = image_uri(buffer.tobytes(), "image/jpeg")
        
        self.sense_buffer.push(
            "observation",
            f" I SAW SOMETHING NEW! I WONDER WHAT THAT IS?",
            metadata={
                "level": event.level,
                "data": frame_uri,
            },
        )

    def _save_inspect(self, event: CamEvent) -> None:
        """Persist the novel frame for the future moment store (and debugging). Best-effort."""
        try:
            self.inspect_dir.mkdir(parents=True, exist_ok=True)
            name = f"{time.strftime('%Y%m%d_%H%M%S')}_z{event.novelty_z:.0f}.jpg"
            cv2.imwrite(str(self.inspect_dir / name), event.frame)
        except Exception as e:
            logger.warning(f"Subconscious: failed to save inspect frame: {e}")

    async def stop(self) -> None:
        """Stop the gate, persist the drifting-normal bank for next boot, release the camera."""
        self._stop.set()
        try:
            self.vision_detector.save()
        except Exception as e:
            logger.warning(f"Subconscious: failed to persist recognition bank: {e}")
            
        self.vision.release()
        logger.debug("Subconscious: stopped.")
