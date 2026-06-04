"""Reusable pieces of the two-timescale novelty harness.

The cognition lives here, each piece self-contained:
  features       — the pure-numpy novelty scorers (patch_novelty L1, embed_novelty L2) + frame/vector
                   helpers (to_jpeg, as_vector). Representations come from the shared EmbeddingEngine.
  camera_buffer  — the threaded camera frame source (CameraBuffer)
  fifo           — L1 habituation: SensoryMemory, perceptual novelty over a recency ring
  recognition    — L2 recognition: RecognitionMemory, the lifelong NORMAL-embedding bank + its seeding
  detector       — NoveltyDetector: L1 + L2 + peak-capture, the async observe(frame, now) -> CamView step

Dependencies flow down: EmbeddingEngine <- {features, recognition, detector}; features <- {fifo (L1),
recognition (L2)} <- detector; L1 and L2 never depend on each other."""
