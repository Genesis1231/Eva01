import select
import sys
import termios
import time
import tty
from datetime import datetime
from pathlib import Path
from typing import Optional

import soundfile as sf

from config import logger
from eva.utils.stt.mic import Microphone
from eva.utils.stt.transcriber import Transcriber

_SPACE = 0x20
_ESC = 0x1B
_PRESS_TIMEOUT_S = 30.0   # seconds to wait for initial space press
_RELEASE_SILENCE_S = 0.15  # seconds of silence that signals key release


class PCListener:
    """Push-to-talk listener for PC/Laptop/WSL.
    Hold SPACE to record audio; release SPACE to transcribe.
    """

    def __init__(self, 
                 model_name: str = "faster-whisper", 
                 language: str = "en",
    ) -> None:
        self.microphone = Microphone()
        self.transcriber = Transcriber(model_name, language)

    def listen(self, save_file: Optional[str] = None) -> tuple[Optional[str], Optional[str]]:
        """Record while SPACE is held, then transcribe and return (text, language)."""
        
        self._print_status("Hold SPACE to talk, release to send.")

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        recording_started = False
        audio_data = None

        try:
            tty.setraw(fd)

            if not self._wait_for_space_press():
                logger.warning("PCListener: no space press within timeout")
                return None, None

            if not self.microphone.start_recording():
                logger.error("PCListener: microphone failed to start")
                return None, None
            recording_started = True

            self._print_status("Recording... release SPACE to send")
            self._wait_for_space_release()

        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            if recording_started:
                audio_data = self.microphone.stop_recording()

        print("\033[K", end="", flush=True)  # clear the status line

        if audio_data is None:
            logger.warning("PCListener: recording too short or failed")
            return None, None

        content, language = self.transcriber.transcribe(audio_data)
        if not content:
            logger.warning("PCListener: no speech detected")
            return None, None

        if save_file:
            path = str(self._data_path() / f"{save_file}.wav")
            sf.write(path, audio_data, samplerate=16000)
            logger.info(f"PCListener: audio saved to {path}")

        return content, language

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _wait_for_space_press(self) -> bool:
        """Block until SPACE is pressed (True) or timeout/ESC (False)."""
        deadline = time.monotonic() + _PRESS_TIMEOUT_S
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if select.select([sys.stdin], [], [], min(0.1, remaining))[0]:
                byte = sys.stdin.buffer.read(1)[0]
                if byte == _SPACE:
                    return True
                if byte == _ESC:
                    return False
        return False

    def _wait_for_space_release(self) -> None:
        """Block until SPACE is released.

        Terminals don't emit key-release events; a held key sends repeated
        bytes instead. We drain those bytes and exit once input goes silent
        for _RELEASE_SILENCE_S seconds.
        """
        while select.select([sys.stdin], [], [], _RELEASE_SILENCE_S)[0]:
            sys.stdin.buffer.read(1)

    def _print_status(self, msg: str) -> None:
        print(f"({datetime.now().strftime('%H:%M:%S')}) {msg}", end="\r", flush=True)

    def _data_path(self) -> Path:
        return Path(__file__).resolve().parents[3] / "data" / "voids"
