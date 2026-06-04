"""L1 habituation / episodic memory: a rolling ring of the last BANK_FRAMES frames.

Every frame is scored vs that memory (perceptual_novelty), then rolling-z-scored over a recent window.
Recency eviction IS the point (habituation) — NOT coverage, which is exactly why L1 is NOT a coreset
(coverage over a lifetime lives in recognition, the OPPOSITE memory). Independent of L2: it only
depends on the shared scorer in features."""

from collections import deque
import numpy as np
from .features import patch_novelty


class SensoryMemory:
    """L1 habituation, recency FIFO: scores each frame's perceptual novelty against a rolling memory of the last
    `bank_frames` frames, then rolling-z-scores it. Recency eviction IS the point (habituation)."""

    BANK_FRAMES = 120          # number of frames to keep in the recency memory.
    WARMUP_FRAMES = 30         # number of frames to fill the memory before scoring (novelty=0.0, scored=False until then)
    Z_WINDOW = 120             # number of frames for rolling z-score (smaller = more responsive, but noisier)

    def __init__(self):
        self.memory: np.ndarray | None = None  
        self.written_frames = 0
        self.history = deque(maxlen=self.Z_WINDOW)

    def observe(self, patches: np.ndarray) -> tuple[float, float, bool] | None:
        """Score `patches` vs the recency memory, then admit them (FIFO eviction). Returns
        (perceptual_novelty, novelty_z, scored); novelty/z stay 0.0 and scored=False until the memory
        and the z-window have filled past WARMUP. Returns None if the grid size differs."""

        if self.memory is None: # initialize memory on the first call
            self.patches_per_frame = patches.shape[0]
            self.memory = np.zeros((self.BANK_FRAMES * self.patches_per_frame, patches.shape[1]), np.float32)

        if patches.shape[0] != self.patches_per_frame:
            return None

        filled_frames = min(self.written_frames, self.BANK_FRAMES)
        perceptual_novelty = novelty_z = 0.0
        scored = False
        
        if filled_frames >= self.WARMUP_FRAMES:  
            # enough history to score (novelty=0.0, scored=False until then)
            perceptual_novelty = patch_novelty(patches, self.memory[:filled_frames * self.patches_per_frame])
            if len(self.history) >= self.WARMUP_FRAMES:
                novelty_z = float((perceptual_novelty - np.mean(self.history)) / (np.std(self.history) + 1e-9))
                scored = True
            self.history.append(perceptual_novelty)

        slot = self.written_frames % self.BANK_FRAMES
        self.memory[slot * self.patches_per_frame:(slot + 1) * self.patches_per_frame] = patches
        self.written_frames += 1
        
        return perceptual_novelty, novelty_z, scored
