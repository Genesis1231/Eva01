"""
AudioSense — EVA's ears.

Supports two push-to-talk trigger modes:
  - keyboard: SPACE key triggers mic recording (PC/laptop)
  - websocket: caller pushes raw audio via receive_audio() (frontend/mobile)

Both paths feed the same internal queue → transcribe → SenseBuffer.
"""

import os
import queue
import select
import signal
import sys
import termios
import threading
import time
import tty
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Callable

import numpy as np

from config import logger
from .mic import Microphone
from .speaker_identifier import SpeakerIdentifier
from .transcriber import Transcriber
from ..sense_buffer import SenseBuffer


class AudioSense:
    """Background audio thread.

    Listens for speech via push-to-talk (keyboard SPACE or an external
    receive_audio() call), transcribes, and writes to the shared SenseBuffer.
    """

    _SPACE = 0x20
    _ESC = 0x1B
    _RELEASE_SILENCE_S = 0.6

    def __init__(
        self,
        transcriber: Transcriber,
        speaker_identifier: Optional[SpeakerIdentifier] = None,
        on_interrupt: Optional[Callable[[], None]] = None,
        is_speaking: Optional[Callable[[], bool]] = None,
        on_transcript: Optional[Callable[[str], None]] = None,
    ) -> None:
        """
        Args:
            transcriber: Transcriber instance (model backend already loaded).
            speaker_identifier: Optional SpeakerIdentifier for voice recognition.
            on_interrupt: Optional sync callback to stop current speech.
            is_speaking:  Optional callable returning True if EVA is speaking.
            on_transcript: Optional sync tap fed each transcript (e.g. the
                subconscious SpeechWindow). Called from the process thread.
        """
        self.transcriber = transcriber or Transcriber()
        self._speaker_id = speaker_identifier
        self._on_interrupt = on_interrupt
        self._is_speaking = is_speaking
        self._on_transcript = on_transcript
        self._mic = Microphone()
        self._keyboard = True
        # Audio queues
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stop_event = threading.Event()
        self._input_thread: Optional[threading.Thread] = None
        self._process_thread: Optional[threading.Thread] = None
        self._executor = ThreadPoolExecutor(max_workers=2)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, buffer: SenseBuffer) -> None:
        """Start background threads, writing transcriptions to buffer."""
        if self._process_thread is not None and self._process_thread.is_alive():
            logger.warning("AudioSense: Already running.")
            return

        self._stop_event.clear()

        self._process_thread = threading.Thread(
            target=self._process_loop, args=(buffer,), daemon=True
        )
        self._process_thread.start()

        if self._keyboard:
            self._input_thread = threading.Thread(
                target=self._input_loop, daemon=True
            )
            self._input_thread.start()

        logger.debug(f"AudioSense: Started (keyboard={self._keyboard}).")

    def stop(self) -> None:
        """Stop all threads cleanly."""
        if self._process_thread is None:
            return

        self._stop_event.set()

        if self._input_thread:
            self._input_thread.join(timeout=3)
            self._input_thread = None

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

    def _input_loop(self) -> None:
        """Thread: monitors SPACE key, records mic audio, queues it."""
        print("   ... PRESS SPACE to talk to EVA ...\r", end="", flush=True)
  
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
      
        try:
            tty.setcbreak(fd)

            while not self._stop_event.is_set():
                if self._await_space_press():
                    # Interrupt EVA if she's speaking
                    if self._is_speaking and self._is_speaking():
                        if self._on_interrupt:
                            self._on_interrupt()
                        logger.debug("AudioSense: interrupted speech")

                        # Wait for speech to actually stop and release device
                        for _ in range(20):
                            if not self._is_speaking():
                                break
                            time.sleep(0.1)

                    if not self._mic.start_recording():
                        logger.error("AudioSense: mic recording failed to start")
                        continue

                    print("   ...Recording... RELEASE to send ...\r", end="", flush=True)
                    self._await_space_release()

                    audio = self._mic.stop_recording()
                    print("   ... PRESS SPACE to talk to EVA ...\r", end="", flush=True)

                    if audio is not None:
                        self._audio_queue.put(audio)
                    else:
                        logger.warning("AudioSense: audio was None — too short or empty")

        except Exception as e:
            logger.error(f"AudioSense: input loop error — {e}")
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

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

    # ------------------------------------------------------------------
    # Keyboard helpers
    # ------------------------------------------------------------------

    def _await_space_press(self) -> bool:
        """Non-blocking check for SPACE; handles ESC and Ctrl+C. Returns True if SPACE."""
        if select.select([sys.stdin], [], [], 0.1)[0]:
            byte = sys.stdin.buffer.read(1)[0]
            if byte == self._SPACE:
                return True
            if byte == self._ESC or byte == 0x03:  # ESC or Ctrl+C
                self._stop_event.set()
                os.kill(os.getpid(), signal.SIGINT)
        return False

    def _await_space_release(self) -> None:
        """Block until the SPACE key is released (silence gap)."""
        while select.select([sys.stdin], [], [], self._RELEASE_SILENCE_S)[0]:
            sys.stdin.buffer.read(1)
