"""EVA's running mood — go_emotions probabilities, decayed + EMA-updated.

The mood vector is a 28-dim probability distribution over the
go_emotions label set (SamLowe/roberta-base-go_emotions-onnx). External
sense inputs update it through a decay-then-EMA reducer attached to the
LangGraph state. Inner-voice ("thought") events do not update mood —
outside factors change mood, inner narration doesn't (matches
human-psychology priors).

"""

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from config import DATA_DIR, logger
from eva.subconscious._mood.appraisal import (
    GO_EMOTIONS_LABELS, 
    _appraise,
    detect_direction,
    strip_sense_tag,
)
from eva.subconscious._mood.labels import EMPATHIC_REACTIONS, DIRECTED_REACTIONS
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
    """LangGraph reducer: decay prior toward zero, then EMA-blend new event."""
    
    if not prior:
        # initial mood is flat/neutral 
        prior = [0.0 for _ in GO_EMOTIONS_LABELS]
        
    decayed = [p * DECAY for p in prior]
    return [ALPHA * n + (1 - ALPHA) * d for d, n in zip(decayed, new_probs)]


class MoodScorer:
    """go_emotions mood score + SOUL-conditioned appraisal.  """

    def __init__(self) -> None:
        self.initialize_mood()
        self.soul_profile: list[float] = self.initialize_soul()
        logger.debug("MoodScorer: emotions model ready.")

    def initialize_mood(self) -> None:
        """Load ONNX model + tokenizer, then score SOUL.md as the trait profile."""

        if not (_ONNX_PATH.exists() and _TOKENIZER_PATH.exists()):
            raise FileNotFoundError(
                f"go_emotions model files not found in {_MODEL_DIR}. "
                "Run `eva-setup` to download them."
            )

        try:
            self._tokenizer = Tokenizer.from_file(str(_TOKENIZER_PATH))
            self._tokenizer.enable_truncation(max_length=256)
            self._session = ort.InferenceSession(
                str(_ONNX_PATH),
                providers=["CPUExecutionProvider"],
            )
        except Exception as e:
            logger.error(f"MoodScorer: failed to initialize model - {e}")
            raise RuntimeError(f"MoodScorer initialization failed. {e}") 
    
    def initialize_soul(self) -> list[float]:
        """ Score SOUL.md to get the initial trait profile"""
        return self._raw(load_prompt("SOUL"))

    def _raw(self, text: str) -> list[float]:
        """28 raw probabilities aligned to :data:`GO_EMOTIONS_LABELS`."""

        enc = self._tokenizer.encode(text)
        input_ids = np.asarray([enc.ids], dtype=np.int64)
        attention_mask = np.asarray([enc.attention_mask], dtype=np.int64)
        outputs = self._session.run(None, {
            "input_ids": input_ids, 
            "attention_mask": attention_mask
        })
        logits = np.asarray(outputs[0])[0]
        return self._sigmoid(logits).tolist()

    def score(self, text: str, source: str | None = None) -> list[float]:
        """Score ``text`` with optional speaker→listener appraisal. """

        text = strip_sense_tag(text)
        raw = self._raw(text)
        if source != "audio":
            return raw
        direction = detect_direction(text)
        if direction == "contagion":
            return raw
        direction_table = DIRECTED_REACTIONS if direction == "directed" else EMPATHIC_REACTIONS
        return _appraise(raw, direction_table, self.soul_profile)

    
    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        # Numerically stable sigmoid (avoids overflow for large negative x).
        return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)),
                        np.exp(x) / (1.0 + np.exp(x)))
    
def surfaced_emotions(
    mood: list[float] | None,
    top_k: int = RENDER_TOP_K,
) -> list[tuple[str, float]]:
    """The mood's top-k non-neutral emotions that clear :data:`RENDER_THRESHOLD`,
    strongest first. Shared by :func:`render_mood` (the brain's prompt block) and
    the Room feed's ``mood_labels`` — one place for the filter/sort/threshold."""
    if not mood:
        return []
    pairs = sorted(
        ((label, p) for label, p in zip(GO_EMOTIONS_LABELS, mood)
         if label != "neutral"),
        key=lambda kv: -kv[1],
    )
    return [(label, p) for label, p in pairs[:top_k] if p >= RENDER_THRESHOLD]


def render_mood(mood: list[float] | None) -> str:
    """Render mood as a compact ``<MOOD>label=N% ...</MOOD>`` block. """
    
    surfaced = surfaced_emotions(mood)
    if not surfaced:
        return ""
    body = " ".join(f"{label}={round(p * 100)}%" for label, p in surfaced)
    return f"<MOOD {body}>"
