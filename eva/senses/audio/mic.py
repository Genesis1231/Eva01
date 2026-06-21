"""
Microphone — EVA's open ear: always-on capture, VAD segmentation, and turn-taking.

The mic listens continuously (no push-to-talk): silero VAD (sherpa-onnx) cuts the
stream into utterances that flow to AudioSense's transcribe pipeline. Turn-taking
lives here because the mic is the only component that can hear it:

  floor    `user_speaking` is True while live speech is in the air — the VoiceActor
           holds Eva's own speech until the floor clears (politeness).
  barge-in the user talking OVER Eva cuts her off. With open speakers the mic hears
           her own TTS — and silero calls that speech — so barge-in needs sustained
           energy above an adaptive echo baseline; VAD alone can't tell her voice
           from the user's.

Echo never reaches the brain: segments that overlapped her playback and stayed under
the barge-in bar are her own voice and are discarded before transcription.
"""

import queue
import threading
import time
from collections import deque
from typing import Callable, Optional

import numpy as np
import sherpa_onnx
import sounddevice as sd

from config import logger, DATA_DIR

_SAMPLE_RATE = 16000
_WINDOW = 512                  # silero's window @ 16 kHz (32 ms)
_VAD_MODEL = "silero_vad.onnx" # under data/models/, fetched by eva-setup
_QUEUE_BLOCKS = 256            # bounded capture queue (~15s headroom: TTS synthesis is
                               # GIL-heavy and can starve the listen thread for seconds)

# VAD segmentation — calibrate live
VAD_THRESHOLD = 0.5      # silero speech-probability gate
MIN_SILENCE = 0.8        # s of quiet that closes an utterance (also the floor's hangover)
MIN_SPEECH = 0.25        # s of speech to count at all (blip filter)
MAX_SPEECH = 25.0        # s hard cap per utterance

# Echo-aware barge-in — calibrate live; per-window RMS/expected/bar logged at debug.
# The echo reference is the PLAYBACK SIGNAL itself: we own the PCM leaving the
# speakers, so expected echo = gain x recent playback level, with the gain measured
# at each utterance's start. (A blind mic-side envelope was tried and failed live:
# it can't tell whose sound it's tracking, so the user's own voice ratchets the bar
# up against their next attempt.) When her waveform dips between words, the bar
# follows it down and the user's voice fires almost instantly.
ECHO_FACTOR = 1.5        # barge-in bar: this x the expected echo
BARGE_WINDOW = 15        # recent windows judged for barge-in (~0.5 s)
BARGE_OVER = 8           # fire when this many of them are over the bar (~0.26 s of voice);
                         # word gaps and VAD flickers don't reset the count — density,
                         # not a streak (a flicker mid-shout must not erase the evidence)
TAIL = 0.25              # s of echo tail ignored after her playback stops
PMAX_DECAY = 0.85        # per-window decay of the playback-level peak (~0.3 s ride-out
                         # for speaker-to-mic latency and room reverb)
SEED_WINDOWS = 15        # sounding windows that calibrate the echo gain per utterance
SEED_TIMEOUT = 30        # windows (~1 s): if playback stays inaudible this long, stop
                         # calibrating — her voice doesn't reach this mic, and waiting
                         # would let the USER's first attempt calibrate the bar against itself
MIN_BASELINE = 0.005     # expected-echo floor so silence can't make the bar
                         # hypersensitive to breaths and chair creaks


class Microphone:
    """Always-on listener: capture → VAD utterances → on_utterance; floor + barge-in."""

    def __init__(
        self,
        on_utterance: Callable[[np.ndarray], None],
        is_playing: Optional[Callable[[], bool]] = None,
        on_interrupt: Optional[Callable[[], None]] = None,
        playback_level: Optional[Callable[[], float]] = None,
    ) -> None:
        """
        Args:
            on_utterance: Receives each finished utterance (float32 mono 16 kHz).
                Called from the listen thread.
            is_playing:   Returns True while EVA's voice is physically in the air.
                Synthesis time is clean air — the VoiceActor handles that window.
            on_interrupt: Thread-safe barge-in trigger (stops EVA's playback).
            playback_level: Returns the instantaneous output RMS — the echo
                reference the barge-in detector compares the mic against.
        """
        self._on_utterance = on_utterance
        self._is_playing = is_playing or (lambda: False)
        self._on_interrupt = on_interrupt
        self._playback_level = playback_level or (lambda: 0.0)

        self._blocks: queue.Queue[np.ndarray] = queue.Queue(maxsize=_QUEUE_BLOCKS)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._stream: Optional[sd.InputStream] = None
        self._vad: Optional[sherpa_onnx.VoiceActivityDetector] = None

        # Turn-taking state (listen thread writes, others only read)
        self._user_speaking = False
        self._was_playing = False      # is_playing() on the previous window
        self._dirty = False            # current speech-run overlapped Eva's playback
        self._gain = 0.0               # mic RMS per unit of playback RMS (echo path)
        self._pmax = 0.0               # decaying peak of recent playback level
        self._seed_left = 0            # gain-calibration windows left (set per utterance)
        self._seed_elapsed = 0         # playback windows seen while still calibrating
        self._recent_over: deque = deque(maxlen=BARGE_WINDOW)  # 1 = window over the bar
        self._tail_until = 0.0         # echo tail after playback ends
        self._mute_until = 0.0         # capture muted after a barge-in
        self._last_overflow_warn = 0.0
        self._level_log = 0            # window counter for the ~1 Hz level debug line

    @property
    def user_speaking(self) -> bool:
        """The floor: True while live (non-echo) speech is in the air."""
        return self._user_speaking

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """Open the stream and start the listen thread. False if the VAD is missing."""
        if self._thread is not None and self._thread.is_alive():
            return True
        if not self._init_vad():
            return False

        try:
            self._stream = sd.InputStream(
                samplerate=_SAMPLE_RATE,
                channels=1,
                dtype="float32",     # silero's native input
                blocksize=0,         # let PortAudio/Pulse pick; fixed tiny blocks overflow
                callback=self._on_audio,
            )
            self._stream.start()
        except Exception as e:
            logger.error(f"Microphone: failed to open input stream — {e}")
            self._stream = None
            return False

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        logger.debug("Microphone: open mic listening.")
        return True

    def stop(self) -> None:
        """Stop the listen thread and close the stream."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
            self._thread = None
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.warning(f"Microphone: error closing stream — {e}")
            self._stream = None

    def _init_vad(self) -> bool:
        model_path = DATA_DIR / "models" / _VAD_MODEL
        if not model_path.exists():
            logger.error(
                f"Microphone: VAD model not found at {model_path}. "
                "Run `eva-setup` to download required models."
            )
            return False
        try:
            cfg = sherpa_onnx.VadModelConfig()
            cfg.silero_vad.model = str(model_path)
            cfg.silero_vad.threshold = VAD_THRESHOLD
            cfg.silero_vad.min_silence_duration = MIN_SILENCE
            cfg.silero_vad.min_speech_duration = MIN_SPEECH
            cfg.silero_vad.max_speech_duration = MAX_SPEECH
            cfg.silero_vad.window_size = _WINDOW
            cfg.sample_rate = _SAMPLE_RATE
            self._vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=30)
            return True
        except Exception as e:
            logger.error(f"Microphone: failed to load VAD — {e}")
            return False

    # ------------------------------------------------------------------
    # Capture (sounddevice callback thread — minimum work only)
    # ------------------------------------------------------------------

    def _on_audio(self, indata, frames, time_info, status) -> None:
        if status:
            logger.debug(f"Microphone: stream status — {status}")
        block = indata[:, 0].copy()
        try:
            self._blocks.put_nowait(block)
        except queue.Full:
            # Drop the oldest — stale audio is worse than lost audio, live.
            try:
                self._blocks.get_nowait()
                self._blocks.put_nowait(block)
            except (queue.Empty, queue.Full):
                pass
            now = time.monotonic()
            if now - self._last_overflow_warn > 10:
                self._last_overflow_warn = now
                logger.warning("Microphone: capture queue overflow, dropping audio")

    # ------------------------------------------------------------------
    # Listen loop (own thread): re-chunk to VAD windows, run turn-taking
    # ------------------------------------------------------------------

    def _listen_loop(self) -> None:
        buf = np.empty(0, dtype=np.float32)
        while not self._stop_event.is_set():
            try:
                block = self._blocks.get(timeout=0.5)
            except queue.Empty:
                continue
            buf = np.concatenate((buf, block))
            while len(buf) >= _WINDOW:
                window, buf = buf[:_WINDOW], buf[_WINDOW:]
                try:
                    self._process_window(window)
                except Exception:
                    logger.error("Microphone: window processing error", exc_info=True)

    def _process_window(self, window: np.ndarray) -> None:
        now = time.monotonic()

        # Post-barge-in mute: drop the echo tail entirely, then restart capture clean.
        # _was_playing is cleared too — the barge already covered the contamination,
        # so the normal playback-end tail must NOT re-dirty the user's ongoing speech
        # (it would discard everything they say until they pause).
        if self._mute_until:
            if now < self._mute_until:
                return
            self._mute_until = 0.0
            self._vad.reset()
            self._dirty = False
            self._was_playing = False
            self._user_speaking = False
            self._recent_over.clear()

        eva_playing = self._is_playing()
        if eva_playing and not self._was_playing:
            # New utterance: fresh candidacy and a fresh echo-gain calibration
            # (the volume knob may have moved between utterances).
            self._recent_over.clear()
            self._seed_left = SEED_WINDOWS
            self._seed_elapsed = 0
            self._gain = 0.0
            self._pmax = 0.0
        elif self._was_playing and not eva_playing:
            self._tail_until = now + TAIL       # playback over: ignore the echo tail
        self._was_playing = eva_playing
        in_tail = now < self._tail_until

        if eva_playing or in_tail:
            self._dirty = True                  # this speech-run is echo-contaminated

        self._vad.accept_waveform(window)

        if eva_playing:
            self._track_echo(window, now)

        # Completed utterances: clean ones surface, contaminated ones are her echo.
        while not self._vad.empty():
            segment = np.asarray(self._vad.front.samples, dtype=np.float32)
            self._vad.pop()
            if self._dirty:
                logger.debug("Microphone: discarded echo-contaminated segment")
            else:
                self._on_utterance(segment)

        speaking_now = self._vad.is_speech_detected()
        if not speaking_now and not (eva_playing or in_tail):
            self._dirty = False                 # quiet and clean: the run is over

        # The floor: echo never counts as the user.
        self._user_speaking = speaking_now and not self._dirty

    def _track_echo(self, window: np.ndarray, now: float) -> None:
        """Compare the mic against the expected echo of what's playing right now."""
        rms = float(np.sqrt(np.mean(window ** 2)))
        self._pmax = max(self._playback_level(), self._pmax * PMAX_DECAY)

        # Calibrate the echo-path gain on this utterance's first sounding windows;
        # times out when playback never reaches the mic (echo-free rig) so the bar
        # arms at the floor instead of waiting to learn from the user's own voice.
        if self._seed_left > 0:
            self._seed_elapsed += 1
            if self._pmax > 1e-4 and rms > MIN_BASELINE:
                self._gain = max(self._gain, rms / self._pmax)
                self._seed_left -= 1
            if self._seed_elapsed >= SEED_TIMEOUT:
                self._seed_left = 0
            self._recent_over.clear()
            return

        expected = self._gain * self._pmax              # her echo, predicted from the PCM
        bar = max(expected, MIN_BASELINE) * ECHO_FACTOR
        self._level_log += 1
        if self._level_log >= 31:                      # ~1 Hz during playback
            self._level_log = 0
            logger.debug(f"Microphone: levels rms={rms:.4f} expected={expected:.4f} bar={bar:.4f}")

        if not self._vad.is_speech_detected():
            self._recent_over.append(0)     # flicker: decay through the window, don't erase
            return

        # Density, not a streak: word gaps must not reset the verdict, or short
        # interjections ("Eva, stop!") never fire. ~0.26 s of over-bar voice within
        # the last ~0.5 s cuts her off; her own echo can't reach that density because
        # the bar tracks her actual waveform.
        self._recent_over.append(1 if rms > bar else 0)
        over = sum(self._recent_over)
        if rms > bar:
            logger.debug(
                f"Microphone: barge candidate rms={rms:.4f} bar={bar:.4f} "
                f"over={over}/{BARGE_WINDOW}"
            )
        if over >= BARGE_OVER:
            self._barge(now)

    def _barge(self, now: float) -> None:
        """The user took the floor: cut Eva off, drop the contaminated audio."""
        logger.info("Microphone: barge-in — the user takes the floor")
        self._recent_over.clear()
        self._user_speaking = True
        self._mute_until = now + TAIL
        if self._on_interrupt is not None:
            try:
                self._on_interrupt()
            except Exception as e:
                logger.error(f"Microphone: interrupt callback failed — {e}")
