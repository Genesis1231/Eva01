"""Route-threshold calibration for L2 visual recognition.

The recognition bank answers "how far is this frame from normal?" This module owns the
route threshold for: an initial held-out normal null, then a rolling online
calibration stream so the threshold can follow changes in lighting, room, or camera view.
"""

from collections import deque
import numpy as np
from .features import embed_novelty


class RouteCalibrator:
    """Adaptive route threshold for long-term visual novelty scores."""

    NULL_PERCENTILE = 95
    HELDOUT_FRACTION = 0.2

    WINDOW = 512
    MIN_SCORES = 20
    FLOOR = 0.1

    def __init__(self, threshold: float, scores: np.ndarray | list[float] | None = None):
        self.threshold = threshold

        if scores is not None:
            observed = np.ravel(scores)
            observed = observed[np.isfinite(observed)].tolist()
        else:
            observed = []

        self._scores = deque(observed[-self.WINDOW:], maxlen=self.WINDOW)
        self._refresh()    # no-op below MIN_SCORES — keeps the passed threshold

    @classmethod
    def initialize(
        cls,
        frames: list[np.ndarray],
        random_generator: np.random.Generator,
    ) -> "RouteCalibrator":
        """Calibrate theta from held-out normal frames scored against the build split."""

        if len(frames) <= 1:
            return cls(threshold=float("inf"))

        order = random_generator.permutation(len(frames))

        ideal_holdout = max(3, int(cls.HELDOUT_FRACTION * len(frames)))
        num_holdout = min(ideal_holdout, len(frames) - 1)

        holdout = [frames[i] for i in order[:num_holdout]]
        build = [frames[i] for i in order[num_holdout:]]
        build_rows = np.vstack(build)

        null_scores = np.array(
            [embed_novelty(held_out, build_rows) for held_out in holdout],
            dtype=np.float32,
        )
        threshold = float(np.percentile(null_scores, cls.NULL_PERCENTILE))
        return cls(threshold=threshold, scores=null_scores)

    def observe(self, score: float) -> None:
        """Recalibrate from a presumed-normal pre-gate observation.

        The caller should feed this before the `<= threshold` admit gate. Feeding only
        accepted scores makes the sample left-censored by its own threshold and ratchets
        theta downward over time.
        """

        if np.isfinite(score):
            self._scores.append(float(score))
            self._refresh()

    def _refresh(self) -> None:
        if len(self._scores) >= self.MIN_SCORES:
            self.threshold = max(
                self.FLOOR,
                float(np.percentile(self._scores, self.NULL_PERCENTILE))
            )
