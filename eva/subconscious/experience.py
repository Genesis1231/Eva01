"""Eva's forming moment: an Experience as it's lived, before it settles into a Moment.

A novelty boundary (a vision inspect or speech trigger) opens an experience. It accumulates while
the situation lives (speech into the shared SpeechWindow, the strongest frame, the latest canvas)
and settles at quiescence (QUIET) or a hard age cap (MAX_AGE). encode() then writes one moment:
transcript to txt_key, her words/reflections to the note and note_key, peak frame to img_key, screen
to canvas_key.

It's a class plus a free function on purpose. Experience is pure timing logic (no engine, no DB, now
injected), so the segmentation is unit-testable on its own; encode() is the only I/O (embed and
memorize), kept separate so the store stays LLM-free.
"""

from dataclasses import dataclass

import numpy as np

from config import logger
from eva.core.moment import MomentDB
from ._speech.window import SpeechWindow

QUIET = 60.0            # seconds of no new event before an experience settles
MAX_AGE = 900.0        # ...but settle after this long regardless (a 15-min hard cap)
PRE_ROLL = 20.0        # transcript reaches this far back before the opening trigger (its cause)
NOTE_MAX_CHARS = 400   # her reaction is born gist-sized; cap, don't compress (no LLM this slice)
TXT_MAX_CHARS = 2000   # the transcript cue, bounded, logged on overflow
IMPORTANCE = 0.6       # fixed activation seed (the felt-importance LLM bucket is deferred)


def _ravel(embedding):
    """Flatten a precomputed (1, D) image embedding to (D,) for the vec0 index; None passes through.
    as_vector yields (1, D), and VectorIndex reads the dim from len(), so a (1, D) row would build a
    1-dim index. Flatten first."""
    return None if embedding is None else np.asarray(embedding, dtype=np.float32).ravel()


@dataclass
class Settled:
    """A finished experience, frozen for encoding."""
    lines: list[tuple[float, str, str]]   # (ts, role, text) from the shared window
    peak_embedding: np.ndarray | None     # the strongest inspect frame's row, or None (no sight)
    canvas_embedding: np.ndarray | None   # the latest canvas her actions left, or None (no screen)
    opened_at: float


class Experience:
    """The moment Eva is living right now: opens at a boundary, settles at quiescence."""

    def __init__(self, window: SpeechWindow):
        self._window = window
        self._opened_at: float | None = None
        self._last_event_at: float = 0.0
        self._peak_embedding: np.ndarray | None = None
        self._peak_strength: float = float("-inf")
        self._canvas_embedding: np.ndarray | None = None

    @property
    def is_open(self) -> bool:
        return self._opened_at is not None

    def begin(self, now: float) -> None:
        """A novelty boundary opens an experience. No-op if one is already unfolding."""
        if self._opened_at is None:
            self._opened_at = now
            self._last_event_at = now

    def touch(self, now: float) -> None:
        """Mark 'still happening': any trigger or new speech while open keeps it alive."""
        if self._opened_at is not None:
            self._last_event_at = now

    def mark_peak(self, embedding, strength: float, now: float) -> None:
        """Hold the strongest inspect frame so far (strength = L2 long_nov). Also touches."""
        if self._opened_at is None:
            return
        if strength > self._peak_strength:
            self._peak_strength = strength
            self._peak_embedding = embedding
        self._last_event_at = now

    def mark_canvas(self, embedding, now: float) -> None:
        """Hold the canvas her actions left on screen (latest wins, the screen the experience
        ended on). Also touches."""
        if self._opened_at is None:
            return
        self._canvas_embedding = embedding
        self._last_event_at = now

    def should_settle(self, now: float) -> bool:
        """Settle on quiescence (QUIET since the last event) or the hard MAX_AGE."""
        if self._opened_at is None:
            return False
        return (now - self._last_event_at) > QUIET or (now - self._opened_at) > MAX_AGE

    def settle(self, now: float) -> Settled:
        """Freeze the experience into a Settled record and reset. The transcript reaches PRE_ROLL
        back before the opening trigger, since the words that caused it are already in the window."""
        opened_at = self._opened_at if self._opened_at is not None else now
        settled = Settled(
            lines=self._window.since(opened_at - PRE_ROLL),
            peak_embedding=self._peak_embedding,
            canvas_embedding=self._canvas_embedding,
            opened_at=opened_at,
        )
        self._reset()
        return settled

    def _reset(self) -> None:
        self._opened_at = None
        self._last_event_at = 0.0
        self._peak_embedding = None
        self._peak_strength = float("-inf")
        self._canvas_embedding = None


async def encode(settled: Settled, engine, moments: MomentDB) -> str:
    """Turn a settled experience into one stored Moment. No LLM: the note is her own words, verbatim.
    Each channel is embedded only if present (None-skip, never a constant filler), so an embed
    failure costs a key, not the moment. Returns the new moment id (or '' if there was nothing)."""

    # image (camera) and canvas (screen): precomputed embeddings, flattened for the vec0 index
    img_key = _ravel(settled.peak_embedding)
    canvas_key = _ravel(settled.canvas_embedding)

    # transcript: what was heard and said, the text cue
    spoken = [text for _, role, text in settled.lines if role in ("heard", "me")]
    txt_key = None
    if spoken:
        txt = "\n".join(spoken)
        if len(txt) > TXT_MAX_CHARS:
            logger.debug(f"Experience: transcript {len(txt)} chars over cap, tail-trimming")
            txt = txt[-TXT_MAX_CHARS:]
        txt_key = await engine.embed_one(txt)

    # note: her reaction, her own lines plus felt reflections, verbatim (her words ARE the impression).
    # felt is fed by the (deferred) feel tap; live runs carry only me until that lands.
    reaction = [text for _, role, text in settled.lines if role in ("me", "felt")]
    note = "\n".join(reaction)[-NOTE_MAX_CHARS:]
    note_key = await engine.embed_one(note) if note else None

    if img_key is None and txt_key is None and canvas_key is None and not note:
        logger.debug("Experience: nothing to encode, skipping")
        return ""

    moment_id = await moments.memorize(
        note, img_key, txt_key, importance=IMPORTANCE, canvas_key=canvas_key, note_key=note_key
    )
    logger.debug(
        f"MomentDB: memorized {moment_id}: img={img_key is not None} "
        f"txt={txt_key is not None} canvas={canvas_key is not None} note={bool(note)}"
    )
    return moment_id
