"""EVA's running mood — go_emotions probabilities, decayed + EMA-updated.

The mood vector is a 28-dim probability distribution over the
go_emotions label set (SamLowe/roberta-base-go_emotions-onnx). External
sense inputs update it through a decay-then-EMA reducer attached to the
LangGraph state. Inner-voice ("thought") events do not update mood —
outside factors change mood, inner narration doesn't (matches
human-psychology priors).

Audio events additionally run through a speaker→listener appraisal
(see :mod:`eva.subconscious._mood.appraisal`): the speaker's expressed
emotion is redistributed across the candidates Eva is likely to feel
as a target ("directed") or bystander ("empathic"), weighted by Eva's
SOUL.md trait profile. Vision and tool events skip appraisal and use
pure contagion.

"""

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from config import DATA_DIR, logger
from eva.subconscious._mood.appraisal import (
    DIRECTED_REACTIONS,
    EMPATHIC_REACTIONS,
    _appraise,
    detect_direction,
)
from eva.subconscious._mood.labels import GO_EMOTIONS_LABELS  # re-exported
from eva.utils.prompt import load_prompt


# Update math
DECAY = 0.95   # multiplicative pull toward neutral, applied every turn
ALPHA = 0.2    # responsiveness to new event (EMA weight)

# Rendering thresholds
RENDER_THRESHOLD = 0.10   # skip labels below this in the <MOOD> block
RENDER_TOP_K = 3          # cap on labels surfaced per render

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


class MoodScorer:
    """go_emotions ONNX (CPU, INT8) + SOUL-conditioned appraisal.

    Local-only: fails fast at construction if the model files aren't in
    ``data/models/``. The SOUL profile is computed once at init from
    ``SOUL.md`` and reused for every audio appraisal.
    """

    def __init__(self) -> None:
        self._session = None
        self._tokenizer = None
       
        self.initialize_mood()
        self.soul_profile: list[float] = self.initialize_soul()
        logger.debug("MoodScorer: emotions model + SOUL profile ready.")

    def initialize_mood(self) -> None:
        """Load ONNX model + tokenizer, then score SOUL.md as the trait profile."""

        if not (_ONNX_PATH.exists() and _TOKENIZER_PATH.exists()):
            raise FileNotFoundError(
                f"go_emotions model files not in {_MODEL_DIR}. "
                "Run snapshot_download for "
                "SamLowe/roberta-base-go_emotions-onnx into data/models/."
            )

        self._tokenizer = Tokenizer.from_file(str(_TOKENIZER_PATH))
        self._tokenizer.enable_truncation(max_length=256)
        self._session = ort.InferenceSession(
            str(_ONNX_PATH),
            providers=["CPUExecutionProvider"],
        )
    
    def initialize_soul(self) -> list[float]:
        return self._raw(load_prompt("SOUL"))

    def _raw(self, text: str) -> list[float]:
        """28 raw probabilities aligned to :data:`GO_EMOTIONS_LABELS`."""
        
        if not self._session or not self._tokenizer:
            logger.error("MoodScorer: model not initialized, cannot score mood.")
            return [0.0] * len(GO_EMOTIONS_LABELS)
        
        enc = self._tokenizer.encode(text)
        input_ids = np.asarray([enc.ids], dtype=np.int64)
        attention_mask = np.asarray([enc.attention_mask], dtype=np.int64)
        logits = self._session.run(
            None,
            {
                "input_ids": input_ids, 
                "attention_mask": attention_mask
            },
        )[0][0]
        return self._sigmoid(logits).tolist()

    def score(self, text: str, source: str | None = None) -> list[float]:
        """Score ``text`` with optional speaker→listener appraisal. """
        raw = self._raw(text)
        if source != "audio":
            return raw
        direction = detect_direction(text)
        if direction == "contagion":
            return raw
        table = DIRECTED_REACTIONS if direction == "directed" else EMPATHIC_REACTIONS
        return _appraise(raw, table, self.soul_profile)

    
    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        # Numerically stable sigmoid (avoids overflow for large negative x).
        return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)),
                        np.exp(x) / (1.0 + np.exp(x)))
    
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
