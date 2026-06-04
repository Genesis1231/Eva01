"""L2 long-term recognition — a budget-capped bank of NORMAL pooled embeddings + a route threshold.

Pure STORAGE; representation + scorer are `features.embed` / `features.embed_novelty`. Admission is
score-gated (MemStream, WWW'22): only frames recognised as normal contribute, so anomalies never
poison "normal". When full, a new embedding overwrites a uniformly-random slot — random-replacement
forgetting: an exponential, recency-biased decay (not FIFO's hard cliff), and since recurring normals
are re-admitted they consolidate (more copies = longer survival) while one-offs fade. So L2 is the
SLOW half of a two-timescale forgetting model (L1's recency ring is the fast half): a drifting
"current normal", not a lifelong coverage archive. Salience weighting (protect / boost important
moments) is the eva mainframe's job, not this module's.

Route threshold = held-out null: a bank self-matches its in-sample frames, so a build/calibration
split (not leave-one-out) is the right null — score held-out normal frames vs the bank, take p95."""

from pathlib import Path
import numpy as np

from config import logger
from .features import as_vector, embed_novelty

class RecognitionMemory:
    """L2 recognition: a budget-capped bank of NORMAL pooled embeddings + a route threshold."""

    L2_BUDGET = 8192               # max embeddings in the recognition memory
    NUM_PRIOR = 200                # prior-session NORMAL frames that seed the bank
    NULL_PERCENTILE = 95           # route threshold = held-out p95 of normal long_nov
    HELDOUT_FRACTION = 0.2         # fraction of seed frames reserved for calibration

    def __init__(self, rows: np.ndarray, threshold: float, random_generator: np.random.Generator):
        self.threshold = threshold
        self.random_generator = random_generator
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
    
    @classmethod
    async def seed(cls, prior_stream: Path, cache_path: Path, engine) -> "RecognitionMemory":
        """Build the recognition bank from a prior session of NORMAL frames (embedded, cached)."""

        random_generator = np.random.default_rng(0)
        frames = await cls._cached_embeddings(prior_stream, cache_path, engine)

        if not frames:
            logger.warning("WARN: no prior-session frames — starting with an empty recognition bank.")
            return cls(np.empty((0, 0), dtype=np.float32), float("inf"), random_generator)

        if len(frames) < 10:
            logger.warning(f"WARN: tiny seed ({len(frames)} frames) — threshold will be noisy.")
        order = random_generator.permutation(len(frames))
        
        # Guard against small counts: ensure at least 1 holdout, leave the rest for build
        ideal_holdout = max(3, int(cls.HELDOUT_FRACTION * len(frames)))
        num_holdout = min(ideal_holdout, max(0, len(frames) - 1))
        
        holdout = [frames[i] for i in order[:num_holdout]]
        build = [frames[i] for i in order[num_holdout:]] or frames   # never let the bank be empty
        rows = np.vstack(build)

        if len(rows) > cls.L2_BUDGET:
            rows = rows[random_generator.choice(len(rows), cls.L2_BUDGET, replace=False)]

        if holdout:
            null_scores = np.array([embed_novelty(held_out, rows) for held_out in holdout])
            threshold = float(np.percentile(null_scores, cls.NULL_PERCENTILE))
        else:
            # Fallback if we only had 1 sample and couldn't create a held-out split
            threshold = float('inf')

        logger.debug(f"recognition bank: {len(rows)} embeddings, "
                     f"threshold (long_nov > {threshold:.3f} = held-out p{cls.NULL_PERCENTILE})")

        return cls(rows, threshold, random_generator)

    @staticmethod
    async def _cached_embeddings(prior_stream: Path, cache_path: Path, engine) -> list:
        """Pooled embeddings of prior-session NORMAL frames (cached on disk). One (1, D) per frame."""

        cache_path = cache_path.with_suffix(".npy")
        if cache_path.exists():
            cached = np.load(cache_path)
            logger.debug(f"seed: cache hit ({len(cached)} prior-session frames)")
            # re-normalise on load: a float16 cache is rounded + denormalised, so a warm seed would
            # otherwise drift from the float32 cold seed that wrote it.
            return [v / (np.linalg.norm(v) + 1e-9) for v in cached.astype(np.float32)]

        frame_paths = sorted(prior_stream.glob("*.jpg"))
        picked = frame_paths[:: max(1, len(frame_paths) // RecognitionMemory.NUM_PRIOR)][:RecognitionMemory.NUM_PRIOR]
        logger.debug(f"seeding from {len(picked)} prior-session frames...")

        embeddings = []
        for path in picked:
            vector = as_vector(await engine.embed_image(path.read_bytes()))
            if vector is not None:
                embeddings.append(vector)
        if embeddings:
            np.save(cache_path, np.stack(embeddings).astype(np.float32))   # float32: warm seed == cold seed
        return embeddings

    def score(self, query: np.ndarray) -> float:
        """Novelty of `query` (this frame's pooled embedding) vs the bank. 0.0 if the bank is empty —
        the guard is load-bearing: embed_novelty's `reference @ query[0]` throws on empty rows."""
        if self._count == 0:
            return 0.0
        return embed_novelty(query, self.rows)

    def admit(self, query: np.ndarray) -> None:
        """Admit query embedding(s) in-place — no allocation once live."""

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


