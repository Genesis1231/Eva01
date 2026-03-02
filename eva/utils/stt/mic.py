from typing import Optional

from config import logger
import numpy as np
import sounddevice as sd


class Microphone:
    """
    Pure audio capture and processing utility.

    Handles microphone input, recording, and audio format conversion.
    No user interaction logic - focused solely on audio operations.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        min_record_seconds: float = 0.2,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.min_record_seconds = min_record_seconds

        # Recording state
        self._recording = False
        self._frames: list[np.ndarray] = []

    @staticmethod
    def _to_float32_audio(raw_audio: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Convert int16 PCM into normalized float32 in [-1, 1)."""
        if raw_audio is None or raw_audio.size == 0:
            return None
        return raw_audio.astype(np.float32) / 32768.0

    def start_recording(self) -> bool:
        """Start audio recording stream."""
        if self._recording:
            return True

        self._frames = []

        def on_audio(indata, frame_count, time_info, status) -> None:
            """Audio stream callback."""
            if status:
                logger.debug(f"Audio stream status: {status}")
            if self._recording:
                self._frames.append(indata.copy())

        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="int16",
                callback=on_audio,
            )
            self._stream.start()
            self._recording = True
            return True
        except Exception as exc:
            logger.error(f"Failed to start audio recording: {exc}")
            return False

    def stop_recording(self) -> Optional[np.ndarray]:
        """Stop recording and return captured audio."""
        if not self._recording:
            return None

        self._recording = False

        try:
            self._stream.stop()
            self._stream.close()
        except Exception as exc:
            logger.error(f"Error stopping audio stream: {exc}")
            return None

        if not self._frames:
            logger.debug("No audio frames captured")
            return None

        # Concatenate all frames
        pcm = np.concatenate(self._frames, axis=0).flatten()
        duration = len(pcm) / self.sample_rate

        # Check minimum duration
        min_samples = int(self.sample_rate * self.min_record_seconds)
        if pcm.size < min_samples:
            logger.debug(f"Recording too short: {duration:.2f}s < {self.min_record_seconds}s")
            return None

        return self._to_float32_audio(pcm)