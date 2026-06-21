"""
AudioSense — EVA's ears.

Open mic: the Microphone listens continuously and VAD-segments speech into
utterances (no push-to-talk). Turn-taking lives in the mic — barge-in cuts EVA's
playback, the floor signal lets the VoiceActor wait politely. An external caller
can also push raw audio via receive_audio() (gateway path).

Both paths feed the same internal queue → transcribe → SenseBuffer.
"""

import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Callable

import numpy as np

from config import logger
from .mic import Microphone
from .speaker_identifier import SpeakerIdentifier
from .transcriber import Transcriber
from ..sense_buffer import SenseBuffer


class AudioSense:
    """Background audio threads.

    The Microphone's listen thread captures and segments speech; the process
    thread transcribes, identifies the speaker, and writes to the SenseBuffer.
    """

    def __init__(
        self,
        transcriber: Transcriber,
        speaker_identifier: Optional[SpeakerIdentifier] = None,
        on_interrupt: Optional[Callable[[], None]] = None,
        is_playing: Optional[Callable[[], bool]] = None,
        playback_level: Optional[Callable[[], float]] = None,
        on_transcript: Optional[Callable[[str], None]] = None,
    ) -> None:
        """
        Args:
            transcriber: Transcriber instance (model backend already loaded).
            speaker_identifier: Optional SpeakerIdentifier for voice recognition.
            on_interrupt: Optional sync callback to stop current speech (barge-in).
            is_playing:   Optional callable, True while EVA's voice is physically
                in the air (the mic's echo handling keys on sound, not intent).
            on_transcript: Optional sync tap fed each transcript (e.g. the
                subconscious SpeechWindow). Called from the process thread.
        """
        self.transcriber = transcriber or Transcriber()
        self._speaker_id = speaker_identifier
        self._on_transcript = on_transcript

        # Audio queues
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stop_event = threading.Event()
        self._process_thread: Optional[threading.Thread] = None
        self._executor = ThreadPoolExecutor(max_workers=2)

        self._mic = Microphone(
            on_utterance=self._audio_queue.put,
            is_playing=is_playing,
            on_interrupt=on_interrupt,
            playback_level=playback_level,
        )

    @property
    def user_speaking(self) -> bool:
        """The floor: True while live speech is in the air (echo never counts)."""
        return self._mic.user_speaking

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, buffer: SenseBuffer) -> None:
        """Start the open mic and the process thread, writing transcriptions to buffer."""
        if self._process_thread is not None and self._process_thread.is_alive():
            logger.warning("AudioSense: Already running.")
            return

        self._stop_event.clear()

        self._process_thread = threading.Thread(
            target=self._process_loop, args=(buffer,), daemon=True
        )
        self._process_thread.start()

        if self._mic.start():
            print("   ... EVA is listening (open mic) ...\r", end="", flush=True)
        else:
            logger.warning("AudioSense: mic unavailable — running without hearing.")

        logger.debug("AudioSense: Started.")

    def stop(self) -> None:
        """Stop all threads cleanly."""
        if self._process_thread is None:
            return

        self._stop_event.set()
        self._mic.stop()

        self._process_thread.join(timeout=3)
        self._process_thread = None

        self.transcriber.close()
        if self._speaker_id:
            self._speaker_id.close()
        self._executor.shutdown(wait=False)
        logger.debug("AudioSense: Stopped.")

    # ------------------------------------------------------------------
    # External audio ingestion (WebSocket / gateway path)
    # ------------------------------------------------------------------

    def receive_audio(self, audio: np.ndarray) -> None:
        """Push raw audio in from an external source (e.g. WebSocket gateway).

        Safe to call from any thread. Audio must be float32, 16 kHz, mono.
        """
        self._audio_queue.put(audio)

    # ------------------------------------------------------------------
    # Internal threads
    # ------------------------------------------------------------------

    def _process_loop(self, buffer: SenseBuffer) -> None:
        """Thread: drains audio queue, transcribes + identifies speaker in parallel."""
        while not self._stop_event.is_set():
            try:
                audio = self._audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                # Run speaker ID + transcription in parallel
                speaker_future = (
                    self._executor.submit(self._speaker_id.identify, audio)
                    if self._speaker_id else None
                )
                transcription_future = self._executor.submit(
                    self.transcriber.transcribe, audio
                )

                speaker = speaker_future.result() if speaker_future else None
                result = transcription_future.result()

                if result:
                    text, _ = result
                    if speaker and speaker.get("name"):
                        content = f"{speaker['name']} said: {text}"
                    else:
                        content = f"I heard: {text}"

                    metadata = (
                        {"speaker_id": speaker["id"]}
                        if speaker and speaker.get("id")
                        else None
                    )
                    buffer.push("audio", content, metadata=metadata)
                    logger.debug(f"AudioSense: pushed audio text — {content}")

                    if self._on_transcript:
                        try:
                            self._on_transcript(content)
                        except Exception as e:
                            logger.warning(f"AudioSense: transcript tap failed — {e}")
                else:
                    logger.warning("AudioSense: no speech detected")
            except Exception as e:
                logger.error(f"AudioSense: transcription error — {e}")
