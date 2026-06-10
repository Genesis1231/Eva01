"""L2 long-term recognition: a budget-capped bank of NORMAL pooled embeddings.

The representation and scorer are features.embed / features.embed_novelty. Admission is score-gated
(MemStream, WWW'22): only frames recognised as normal contribute, so anomalies never poison
"normal". When the bank is full a new embedding overwrites a uniformly-random slot. That random
replacement is the forgetting: an exponential, recency-biased decay rather than FIFO's hard cliff,
and since recurring normals get re-admitted they consolidate (more copies means longer survival)
while one-offs fade. So L2 is the slow half of a two-timescale forgetting model (L1's recency ring is
the fast half): a drifting "current normal", not a lifelong coverage archive. Salience weighting
(protecting or boosting important moments) is the eva mainframe's job, not this module's.

Route calibration lives in RouteCalibrator: the held-out split is calibration-only, and all normal
seed frames still rejoin this bank."""

from pathlib import Path

import numpy as np

from config import logger
from .calibration import RouteCalibrator
from .features import as_vector, embed_novelty


class RecognitionMemory:
    """The L2 bank (budget-capped NORMAL embeddings): score(frame) gives novelty vs normal,
    admit(v) grows it (score-gated, random replacement), and seed()/save() persist it."""

    L2_BUDGET = 8192               # max embeddings in the recognition memory
    NUM_PRIOR = 200                # prior-session NORMAL frames that seed the bank

    def __init__(
        self,
        rows: np.ndarray,
        threshold: float,
        random_generator: np.random.Generator,
        cache_path: Path,
        calibrator: RouteCalibrator | None = None,
    ):
        self.calibrator = calibrator or RouteCalibrator(threshold)
        self.random_generator = random_generator
        self.cache_path = cache_path

        self._count = len(rows)
        if self._count:
            self._buffer = np.empty((self.L2_BUDGET, rows.shape[1]), dtype=np.float32)
            self._buffer[:self._count] = rows
        else:
            self._buffer = None    # embedding dim unknown until first admit

    @property
    def rows(self):
        if self._buffer is None or self._count == 0:
            return np.empty((0, 0), dtype=np.float32)
        return self._buffer[:self._count]

    @property
    def count(self):
        return self._count

    @property
    def threshold(self) -> float:
        return self.calibrator.threshold

    @classmethod
    async def seed(cls, prior_stream: Path, cache_path: Path, engine) -> "RecognitionMemory":
        """Build the recognition bank from a prior session of NORMAL frames (embedded, cached)."""

        random_generator = np.random.default_rng(0)
        frames = await cls._cached_embeddings(prior_stream, cache_path, engine)

        if not frames:
            logger.warning("WARN: no prior-session data, starting with an empty recognition bank.")
            return cls(
                rows=np.empty((0, 0), dtype=np.float32),
                threshold=float("inf"),
                random_generator=random_generator,
                cache_path=cache_path
            )

        if len(frames) < 20:
            logger.warning(f"WARN: tiny seed ({len(frames)} frames), threshold will be noisy.")

        calibrator = RouteCalibrator.initialize(frames, random_generator)

        rows = np.vstack(frames)
        if len(rows) > cls.L2_BUDGET:
            rows = rows[random_generator.choice(len(rows), cls.L2_BUDGET, replace=False)]

        logger.debug(f"recognition bank: {len(rows)} embeddings, "
                     f"threshold (long_nov > {calibrator.threshold:.3f} = "
                     f"held-out p{RouteCalibrator.NULL_PERCENTILE})")

        return cls(
            rows=rows,
            threshold=calibrator.threshold,
            random_generator=random_generator,
            cache_path=cache_path,
            calibrator=calibrator
        )

    @staticmethod
    async def _cached_embeddings(prior_stream: Path, cache_path: Path, engine) -> list:
        """Pooled embeddings of prior-session NORMAL frames (cached on disk). One (1, D) per frame."""

        cache_path = cache_path.with_suffix(".npy")
        if cache_path.exists():
            cached = np.load(cache_path).astype(np.float32)
            logger.debug(f"seed: cache hit ({len(cached)} prior-session frames)")
            # Tolerate both on-disk layouts (legacy (N, 1, D) from np.stack, and the (N, D) that
            # save() writes) and yield (1, D) rows, which is what seed/embed_novelty expect. Re-normalise
            # per row: a float16 cache is rounded/denormalised, so a warm seed would otherwise drift.
            rows = cached.reshape(cached.shape[0], -1)
            # Normalize the whole matrix at once along axis 1
            rows = rows / (np.linalg.norm(rows, axis=1, keepdims=True) + 1e-9)

            return [row[None, :] for row in rows]

        frame_paths = sorted(prior_stream.glob("*.jpg"))
        # Evenly space indices and pick them
        indices = np.linspace(0, len(frame_paths) - 1, min(len(frame_paths), RecognitionMemory.NUM_PRIOR), dtype=int)
        picked = [frame_paths[i] for i in indices]
        logger.debug(f"seeding from {len(picked)} prior-session frames...")

        embeddings = []
        for path in picked:
            vector = as_vector(await engine.embed_image(path.read_bytes()))
            if vector is not None:
                embeddings.append(vector)
        
        if embeddings:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, np.vstack(embeddings).astype(np.float32))  # (N, D); warm seed == cold seed
        
        return embeddings

    def score(self, frame: np.ndarray) -> float:
        """Novelty of this frame's pooled embedding vs the bank. Higher is more novel."""
        if self._count == 0:
            return 0.0
        return embed_novelty(frame, self.rows)

    def observe_calibration_score(self, score: float) -> None:
        """Observe a score for online route threshold calibration."""
        self.calibrator.observe(score)

    def save(self) -> None:
        """Persist the current bank as the next session's warm-seed cache."""
        if self._count:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(self.cache_path, self.rows.astype(np.float32))

    def admit(self, query: np.ndarray) -> None:
        """Admit query embedding(s) in place, with random-choice replacement when the bank is full."""

        n_new = len(query)
        if self._buffer is None:
            self._buffer = np.empty((self.L2_BUDGET, query.shape[1]), dtype=np.float32)

        remaining = self.L2_BUDGET - self._count
        if n_new <= remaining:
            self._buffer[self._count:self._count + n_new] = query
            self._count += n_new
        else:
            if remaining > 0:
                self._buffer[self._count:] = query[:remaining]
                self._count = self.L2_BUDGET
                query = query[remaining:]

            n = min(len(query), self.L2_BUDGET)
            idxs = self.random_generator.choice(self.L2_BUDGET, n, replace=False)
            self._buffer[idxs] = query[:n]
