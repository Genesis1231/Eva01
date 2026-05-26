"""EVA's running mood — go_emotions probabilities, decayed + EMA-updated.

The mood vector is a 28-dim probability distribution over the
go_emotions label set (SamLowe/roberta-base-go_emotions-onnx). External
sense inputs (audio, vision, tool) update it through a decay-then-EMA
reducer attached to the LangGraph state. Inner-voice ("thought") events
do not update mood — outside factors change mood, inner narration
doesn't (matches human-psychology priors).

Inference runs directly on ``onnxruntime`` with the Rust-backed
``tokenizers`` library — no ``optimum`` or ``transformers`` dependency,
avoiding the ~500MB torch / sentencepiece chain. Total runtime cost is
the ONNX session + ~10MB Rust tokenizer.

The state lives at :attr:`EvaState.mood` in :mod:`eva.core.graph`.
Rendering for the system prompt is :func:`render_mood`.
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from config import DATA_DIR, logger


# Labels in the model's emission order — taken from config.json
# (SamLowe/roberta-base-go_emotions-onnx). The i-th probability the
# model emits corresponds to the i-th label here, so this list is also
# the canonical index → name mapping for the mood vector.
GO_EMOTIONS_LABELS: list[str] = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral",
]

# Update math
DECAY = 0.95   # multiplicative pull toward neutral, applied every turn
ALPHA = 0.2    # responsiveness to new event (EMA weight)

# Rendering thresholds
RENDER_THRESHOLD = 0.10   # skip labels below this in the <MOOD> block
RENDER_TOP_K = 5          # cap on labels surfaced per render

# Inference limits — mirrors the previous transformers pipeline call
# (truncation=True, max_length=256).
_MAX_TOKENS = 256

_MODEL_DIR = DATA_DIR / "models"
_ONNX_PATH = _MODEL_DIR / "onnx" / "model_quantized.onnx"
_TOKENIZER_PATH = _MODEL_DIR / "tokenizer.json"


def _update_mood(
    prior: list[float] | None,
    new_probs: list[float],
) -> list[float]:
    """LangGraph reducer: decay prior toward zero, then EMA-blend new event.

    First event (no prior) is taken verbatim. Subsequent events nudge the
    running probabilities; quiet periods (between events) don't decay
    here — that's deferred to a future maintenance heartbeat tick.
    """
    if not prior:
        return list(new_probs)
    decayed = [p * DECAY for p in prior]
    return [ALPHA * n + (1 - ALPHA) * d for d, n in zip(decayed, new_probs)]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid (avoids overflow for large negative x).
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


class MoodScorer:
    """go_emotions ONNX (CPU, INT8) — pure onnxruntime + tokenizers path.

    Local-only: fails fast at construction if the model files aren't in
    ``data/models/``. Mirrors the kokoro / silero_vad loader pattern.
    """

    def __init__(self) -> None:
        self._session = None
        self._tokenizer = None
        
        self.initialize_mood()
        logger.debug("MoodScorer: emotions model ready.")

    def initialize_mood(self) -> None:
        """Explicit model loading from construction."""
        
        if not (_ONNX_PATH.exists() and _TOKENIZER_PATH.exists()):
            raise FileNotFoundError(
                f"go_emotions model files not in {_MODEL_DIR}. "
                "Run snapshot_download for "
                "SamLowe/roberta-base-go_emotions-onnx into data/models/."
            )

        self._tokenizer = Tokenizer.from_file(str(_TOKENIZER_PATH))
        self._tokenizer.enable_truncation(max_length=_MAX_TOKENS)
        self._session = ort.InferenceSession(
            str(_ONNX_PATH),
            providers=["CPUExecutionProvider"],
        )        
        
        
    def score(self, text: str) -> list[float]:
        """Return 28 probabilities aligned to :data:`GO_EMOTIONS_LABELS`."""
        enc = self._tokenizer.encode(text)
        input_ids = np.asarray([enc.ids], dtype=np.int64)
        attention_mask = np.asarray([enc.attention_mask], dtype=np.int64)
        logits = self._session.run(
            None,
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )[0][0]
        return _sigmoid(logits).tolist()


def render_mood(mood: list[float] | None) -> str:
    """Render mood as a compact ``<MOOD>label=N% ...</MOOD>`` block.

    Returns the empty string when mood is genuinely flat (no label clears
    :data:`RENDER_THRESHOLD`) — no block, no noise. EVA sees the raw
    probabilities and articulates them in her own voice (or doesn't).
    """
    if not mood:
        return ""
    pairs = sorted(
        ((label, p) for label, p in zip(GO_EMOTIONS_LABELS, mood)
         if label != "neutral"),
        key=lambda kv: -kv[1],
    )
    surfaced = [
        (label, p) for label, p in pairs[:RENDER_TOP_K]
        if p >= RENDER_THRESHOLD
    ]
    if not surfaced:
        return ""
    body = " ".join(f"{label}={round(p * 100)}%" for label, p in surfaced)
    return f"<MOOD {body}>"
