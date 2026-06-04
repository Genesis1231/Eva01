"""SpeechDetector — L1 recent-novelty over the agent-state text (the speech channel's trigger).

Mirrors _vision's L1: embed the current agent-state text, score how different it is from the last
RECENT text vectors (kNN cosine), rolling-z + floor -> a candidate trigger. There is NO L2 here —
"have I heard this before?" is MomentDB.recall (the shared, cross-modal recognition layer). Each
observe() is one utterance (speech is already discrete), so no peak-capture/refractory — it returns
the per-event reading plus the txt_key a moment is cued by.

Driven on each new transcript by the subconscious processor (not wired yet, parallel to _vision).
Constants need live calibration — text novelty runs on a higher scale than patch novelty."""

from collections import deque
from dataclasses import dataclass

import numpy as np

from .window import SpeechWindow, SENTINEL

RECENT = 40                            # recent text vectors kept for habituation (~minutes of talk)
WARMUP = 5                             # vectors needed before scoring
Z_WINDOW = 40                          # rolling window for the z-score
KNN = 8                                # per-utterance k-NN smoothing
Z_TRIGGER, NOVELTY_FLOOR = 2.0, 0.5    # text novelty runs higher than patches → higher floor; calibrate live


@dataclass(eq=False)
class SpeechView:
    """What observe() returns for an utterance: the L1 reading + the txt_key it embeds to."""
    novelty: float
    novelty_z: float
    triggered: bool
    text: str               # the agent-state text that was scored
    txt_key: list[float]    # its embedding — the moment's text cue


class SpeechDetector:
    """L1 recent-novelty over the agent-state text. Recognition is MomentDB's job, not L2's."""

    def __init__(self, window: SpeechWindow, engine):
        self.window = window
        self.engine = engine
        self._recent: deque[np.ndarray] = deque(maxlen=RECENT)
        self._history: deque[float] = deque(maxlen=Z_WINDOW)

    async def observe(self) -> SpeechView | None:
        """Embed the current agent-state text and score its recent-novelty. None when idle ('quiet')
        or the embed failed — so the caller can skip cheaply."""
        text = self.window.current()
        if text == SENTINEL:
            return None

        embedding = await self.engine.embed_one(text)
        if embedding is None:
            return None

        vector = np.asarray(embedding, np.float32)
        vector /= np.linalg.norm(vector) + 1e-9

        novelty, novelty_z, scored = self._score(vector)
        self._recent.append(vector)                # admit AFTER scoring (no self-match)
        triggered = scored and novelty_z > Z_TRIGGER and novelty > NOVELTY_FLOOR
        return SpeechView(novelty, novelty_z, triggered, text, embedding)

    def _score(self, vector: np.ndarray) -> tuple[float, float, bool]:
        """Recent-novelty = 1 - mean of the KNN nearest cosines to recent text; rolling-z, warmup-gated.
        novelty/z stay 0.0 (scored=False) until the recent bank and z-window have filled past WARMUP."""
        if len(self._recent) < WARMUP:
            return 0.0, 0.0, False

        similarities = np.stack(self._recent) @ vector       # recent + query are unit-norm → cosine
        k = min(KNN, similarities.shape[0])
        novelty = float(1.0 - np.partition(similarities, -k)[-k:].mean())

        novelty_z, scored = 0.0, False
        if len(self._history) >= WARMUP:
            novelty_z = float((novelty - np.mean(self._history)) / (np.std(self._history) + 1e-9))
            scored = True
        self._history.append(novelty)
        return novelty, novelty_z, scored
