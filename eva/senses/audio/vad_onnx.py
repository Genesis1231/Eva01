"""
SileroVAD — pure onnxruntime + numpy voice activity detector.

Replaces the silero-vad pip package (which depends on torch) with a thin wrapper
around the official silero_vad.onnx export. Same public contract as the pip
package's `get_speech_timestamps`: returns a list of `{"start": int, "end": int}`
spans in sample indices.

Silero v5 @ 16 kHz processes the audio in 512-sample chunks (32 ms each).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import onnxruntime as ort

from config import DATA_DIR, logger

def pick_providers() -> List[str]:
    """ONNX execution providers ordered by preference for this platform."""
    if sys.platform == "darwin":
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    
    return ["CUDAExecutionProvider", "CPUExecutionProvider"]

class SileroVAD:
    """Silero VAD v5 on pure onnxruntime."""

    CHUNK_SIZE = 512           # Silero v5 @ 16 kHz = 32 ms per chunk
    SAMPLE_RATE = 16000
    THRESHOLD = 0.5            # speech probability threshold
    _CONTEXT_SIZE = 64         # last 64 samples of previous chunk, prepended
    _MIN_SPEECH_MS = 250       # drop spurious blips
    _MIN_SILENCE_MS = 100      # min gap before splitting a segment
    _SPEECH_PAD_MS = 30        # pad each detected span on both sides

    def __init__(self):
        self._session = self.initialize_ort()
        self._state: np.ndarray = np.zeros((2, 1, 128), dtype=np.float32)
        self._context: np.ndarray = np.zeros((1, self._CONTEXT_SIZE), dtype=np.float32)
        self._sr_arg = np.array(self.SAMPLE_RATE, dtype=np.int64)
        active = self._session.get_providers()[0]
        
        logger.debug(f"SileroVAD model loaded ({active})")

    def initialize_ort(self) -> ort.InferenceSession:
        """Initialize the ONNX Runtime session with the Silero VAD model."""
        
        silero_model_path = Path(DATA_DIR / "models" / "silero_vad.onnx")
        if not silero_model_path.exists():
            raise RuntimeError(
                f"Silero VAD model not found at {silero_model_path}. "
                "Please run `eva-setup` from the command line to download required models."
            )
        
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        
        return ort.InferenceSession(
            str(silero_model_path), sess_options=opts, providers=pick_providers()
        )
        
    def reset_states(self) -> None:
        """ Reset the model's internal state. Call between utterances when using `probe()`"""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self._CONTEXT_SIZE), dtype=np.float32)

    def probe(self, chunk: np.ndarray) -> float:
        """Stream a single 512-sample chunk through the model, return speech prob.

        Maintains state across calls — use `reset_states()` between utterances.
        For batch detection of a whole clip, use `get_speech_timestamps()`.
        """
        if len(chunk) != self.CHUNK_SIZE:
            raise ValueError(f"chunk must be {self.CHUNK_SIZE} samples, got {len(chunk)}")

        chunk = chunk.astype(np.float32, copy=False).reshape(1, -1)
        x = np.concatenate([self._context, chunk], axis=1)
        out, new_state = self._session.run(
            None, {"input": x, "state": self._state, "sr": self._sr_arg}
        )
        self._state = new_state
        self._context = chunk[:, -self._CONTEXT_SIZE:]
        return float(out[0, 0])

    def get_speech_timestamps(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> List[dict]:
        """Detect speech spans in `audio`.

        Args:
            audio: 1-D float32 numpy array, mono, values in [-1, 1].
            sample_rate: must be 16000 (Silero v5 16 kHz model).

        Returns:
            List of {"start": int, "end": int} in sample indices.
        """
        if sample_rate != self.SAMPLE_RATE:
            raise ValueError(f"SileroVAD expects {self.SAMPLE_RATE} Hz, got {sample_rate}")

        self.reset_states()
        n = len(audio)
        probs = [
            self.probe(audio[i : i + self.CHUNK_SIZE])
            for i in range(0, n - self.CHUNK_SIZE + 1, self.CHUNK_SIZE)
        ]
        return self._collect_segments(probs, n)

    def trim_silence(self, audio: np.ndarray, pad_ms: int = 100) -> np.ndarray:
        """Crop leading/trailing silence, padding the kept region by `pad_ms`."""
        stamps = self.get_speech_timestamps(audio, sample_rate=self.SAMPLE_RATE)
        if not stamps:
            return audio

        pad = int(pad_ms * self.SAMPLE_RATE / 1000)
        start = max(0, stamps[0]["start"] - pad)
        end = min(len(audio), stamps[-1]["end"] + pad)
        return audio[start:end]

    def _collect_segments(self, probs: List[float], total_samples: int) -> List[dict]:
        """Walk the per-chunk probabilities, emit speech spans in sample indices."""
        
        min_speech = int(self._MIN_SPEECH_MS * self.SAMPLE_RATE / 1000)
        min_silence = int(self._MIN_SILENCE_MS * self.SAMPLE_RATE / 1000)
        pad = int(self._SPEECH_PAD_MS * self.SAMPLE_RATE / 1000)

        segments: List[dict] = []
        in_speech = False
        cur_start = 0
        silence_run = 0

        for i, p in enumerate(probs):
            sample_idx = i * self.CHUNK_SIZE
            if p >= self.THRESHOLD:
                if not in_speech:
                    cur_start = sample_idx
                    in_speech = True
                silence_run = 0
            elif in_speech:
                silence_run += self.CHUNK_SIZE
                if silence_run >= min_silence:
                    end = sample_idx - silence_run + self.CHUNK_SIZE
                    if end - cur_start >= min_speech:
                        segments.append({"start": cur_start, "end": end})
                    in_speech = False
                    silence_run = 0

        # Tail: ran off the end while still in speech.
        if in_speech:
            end = len(probs) * self.CHUNK_SIZE
            if end - cur_start >= min_speech:
                segments.append({"start": cur_start, "end": end})

        # Apply padding, clipped to [0, total_samples].
        for seg in segments:
            seg["start"] = max(0, seg["start"] - pad)
            seg["end"] = min(total_samples, seg["end"] + pad)

        return segments

