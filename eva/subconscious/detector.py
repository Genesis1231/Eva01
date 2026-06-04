"""
The visual-novelty detector — ONE patch scorer over TWO memories (NGU style), as a single class.

    L1 = recency FIFO (fifo.SensoryMemory): patch novelty vs the last frames -> z + floor -> candidate
        pop. Recency eviction is habituation — it decides WHEN something is worth a second look.
    L2 = long-term recognition (recognition.RecognitionMemory): on a pop, embed the peak frame and
        measure its distance to the "normal world" -> is this content genuinely NEW?

Detect (L1, every frame) -> Value (L2 graded long_nov on a pop) -> Route (recognised -> "acknowledge"
+ grow the bank; new -> "inspect"). Every committed burst is emitted; deduping redundant re-fires of
a lingering event is the consumer's (eva mainframe's) job — it has the context to weigh salience.

`observe(frame, now)` is the cognitive step (async — it awaits the engine's patch/embed calls). A
subconscious processor or the CLI runner drives it in a camera loop. Each frame yields a CamView; a
finished burst yields a CamEvent. Both representations come from the shared EmbeddingEngine."""

from dataclasses import dataclass
import numpy as np
from pathlib import Path

from config import logger
from ._vision.features import as_vector, to_jpeg
from ._vision.fifo import SensoryMemory
from ._vision.recognition import RecognitionMemory

Z_TRIGGER, NOVELTY_FLOOR = 2.0, 0.25        # pop when rolling z > Z_TRIGGER and novelty > floor
REFRACTORY, PEAK_FALLOFF, PEAK_MAX_HOLD = 5.0, 0.6, 4.0   # REFRACTORY: cooldown measured from a burst CLOSE
TRICKLE_Z, TRICKLE_MIN_GAP = 0.75, 30.0     # grow the bank with "normal but non-trivial" frames
ROUTE = {2: "inspect", 1: "acknowledge"}           # route, don't suppress — both are real events


@dataclass(eq=False)
class CamEvent:
    """A committed novelty event: a burst's peak frame, valued (long_nov) + routed (level) by L2."""
    novelty: float          # L1 perceptual novelty at the peak
    novelty_z: float
    long_nov: float         # L2 distance to the normal world (nan if the embed call failed)
    level: int              # 2 = inspect (new), 1 = acknowledge (recognised)
    route: str
    frame: np.ndarray       # the peak frame — an OWNED copy (snapshot at capture); safe to retain/mutate
    time: float


@dataclass(eq=False)
class CamView:
    """What observe() returns each frame: the per-frame L1 reading + a CamEvent when a burst commits."""
    novelty: float
    novelty_z: float
    triggered: bool
    event: CamEvent | None = None


@dataclass(eq=False)
class _Peak:
    """Internal: the strongest frame of an in-progress trigger-burst, tracked until it fades."""
    novelty: float
    novelty_z: float
    frame: np.ndarray
    time: float


class NoveltyDetector:
    """The visual-novelty detector — ONE patch scorer over TWO memories (NGU style)"""

    def __init__(self, l1: SensoryMemory, l2: RecognitionMemory, engine):
        self.l1 = l1
        self.l2 = l2
        self.engine = engine
        self._peak: _Peak | None = None
        self._last_pop = 0.0
        self._last_trickle = 0.0

    @classmethod
    async def seed(cls, prior_stream: Path, cache: Path, engine) -> "NoveltyDetector":
        """Build a detector whose L2 bank is seeded from a prior session of NORMAL frames."""
        l1 = SensoryMemory()
        l2 = await RecognitionMemory.seed(prior_stream, cache, engine)
        return cls(l1, l2, engine)

    async def observe(self, frame: np.ndarray, now: float) -> CamView | None:
        """Process one frame -> a per-frame CamView (with a CamEvent when a burst commits). Returns
        None only when patch encoding fails, so the caller can time a server-down abort."""

        patches = await self.engine.patch(to_jpeg(frame))
        if patches is None:
            logger.error("Failed to encode patch features for the frame.")
            return None

        scores = self.l1.observe(patches)
        if scores is None:
            logger.error("Patch grid size changed — cannot score novelty. Check the server logs?")
            return CamView(0.0, 0.0, False)

        novelty, novelty_z, scored = scores
        triggered = scored and novelty_z > Z_TRIGGER and novelty > NOVELTY_FLOOR

        event = await self._track_peak(novelty, novelty_z, frame, triggered, now)

        # score-gated trickle: grow recognition with "normal but non-trivial" frames
        if (scored and not triggered and event is None and self._peak is None
                and novelty_z > TRICKLE_Z and now - self._last_trickle > TRICKLE_MIN_GAP):
            vector = as_vector(await self.engine.embed_image(to_jpeg(frame)))
            if vector is not None and self.l2.score(vector) <= self.l2.threshold:
                self.l2.admit(vector)
            self._last_trickle = now

        return CamView(novelty, novelty_z, triggered, event)

    async def flush(self, now: float | None = None) -> CamEvent | None:
        """Commit a peak still in progress. The refractory clock is `now` (the burst close), or the
        peak time when called at shutdown without one."""
        if self._peak is None:
            return None
        close = now if now is not None else self._peak.time
        event = await self._commit(self._peak)
        self._last_pop = close                      # refractory from the CLOSE, not the peak
        self._peak = None
        return event

    # --- peak-capture: commit the STRONGEST frame of a trigger-burst, not the onset ---
    async def _track_peak(self, novelty, novelty_z, frame, triggered, now) -> CamEvent | None:
        """Track a trigger burst: start a peak on trigger, raise it as novelty grows, commit on fade."""
        if self._peak is None:
            if triggered and now - self._last_pop > REFRACTORY:
                self._start_peak(novelty, novelty_z, frame, now)
            return None
        if novelty > self._peak.novelty:
            self._start_peak(novelty, novelty_z, frame, now)
        if self._should_commit(novelty, now):
            return await self.flush(now)
        return None

    def _start_peak(self, novelty, novelty_z, frame, now):
        self._peak = _Peak(novelty, novelty_z, frame.copy(), now)   # own the frame: snapshot it now

    def _should_commit(self, novelty, now) -> bool:
        """Commit when THIS frame has faded below PEAK_FALLOFF of the peak, or the peak has held
        PEAK_MAX_HOLD without being beaten."""

        if not self._peak:
            return False
        return novelty < PEAK_FALLOFF * self._peak.novelty or now - self._peak.time > PEAK_MAX_HOLD

    async def _commit(self, peak: _Peak) -> CamEvent:
        """Value + Route a finished peak: embed it, score vs the bank, route, grow the bank if normal."""

        vector = as_vector(await self.engine.embed_image(to_jpeg(peak.frame)))
        if vector is None:  # embed failed: can't value it -> acknowledge, no admit
            logger.error("Failed to embed the peak frame.")
            return CamEvent(
                peak.novelty,
                peak.novelty_z,
                float("nan"), 1, ROUTE[1],
                peak.frame, peak.time)

        long_nov = self.l2.score(vector)
        level = 2 if long_nov > self.l2.threshold else 1
        if level == 1:                              # recognised normal -> grow the bank (score-gated)
            self.l2.admit(vector)

        return CamEvent(
            peak.novelty,
            peak.novelty_z,
            long_nov, level, ROUTE[level],
            peak.frame, peak.time)
