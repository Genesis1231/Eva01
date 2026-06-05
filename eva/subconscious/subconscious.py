"""EVA's subconscious — the always-on perceptual gate beneath the conscious graph.

Raw frames never enter the SenseBuffer — they stay inside the gate — so they don't reset the Heart's
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
    """The subconscious layer, surfacing novel moments up."""

    GATE_INTERVAL = 1.0     # seconds per glance (~1 fps)
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
        """Beat forever — glance, gate, surface. A peer in wake()'s asyncio.gather."""
        
        if not self.vision.is_available:
            logger.warning("Subconscious: no camera — vision gate disabled.")
            return

        logger.debug(f"Subconscious started. vision gate at ~{1 / self.interval:.1f} fps")
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                await self._glance()
            except Exception as e:
                logger.error(f"Subconscious: gate error — {e}")
                
            await self._pace(self.interval - (time.monotonic() - started))
            # print(".", end="")  # heartbeat for the gate loop

    async def _pace(self, remaining: float) -> None:
        """Sleep the rest of the interval, but wake immediately on stop — holds ~1 fps."""
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=max(0.0, remaining))
        except asyncio.TimeoutError:
            pass
        
    async def _glance(self) -> None:
        """Inspect wakes the brain; acknowledge habituates."""
        
        frame = await asyncio.to_thread(self.vision.capture_photo)
        view = await self.vision_detector.observe(frame, time.monotonic())
        
        if view is None or view.event is None:
            return

        if view.event.level == 2:
            # inspect — genuinely new → wake the brain
            await self._surface(view.event)
        else:
            # acknowledge — familiar → habituate quietly
            logger.debug(f"Subconscious: familiar habituation — novelty_z={view.event.novelty_z:.2f}, long_nov={view.event.long_nov:.3f}")



    async def _surface(self, event: CamEvent) -> None:
        """Describe the novel scene and wake the brain."""
        
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
            logger.warning(f"Subconscious: failed to save inspect frame — {e}")

    async def stop(self) -> None:
        """Stop the gate, persist the drifting-normal bank for next boot, release the camera."""
        self._stop.set()
        try:
            self.vision_detector.save()
        except Exception as e:
            logger.warning(f"Subconscious: failed to persist recognition bank — {e}")
            
        self.vision.release()
        logger.debug("Subconscious: stopped.")
