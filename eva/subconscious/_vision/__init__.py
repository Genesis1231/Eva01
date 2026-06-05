"""The vision channel of Eva's subconscious, a two-timescale novelty detector.

Each piece is self-contained, and the representations themselves (patch tokens and pooled
embeddings) come from the shared EmbeddingEngine rather than from this package:

  features     the pure-numpy novelty scorers, patch_novelty for L1 and embed_novelty for L2,
               together with small frame and vector helpers (to_jpeg, as_vector).
  fifo         L1 habituation, the SensoryMemory recency ring that scores perceptual novelty
               against the most recent frames.
  recognition  L2 recognition, the RecognitionMemory bank of normal embeddings and its seeding,
               which decides whether a flagged frame is genuinely new.
  detector     NoveltyDetector, which combines L1, L2, and peak-capture into one async
               observe(frame, now) step that yields a CamView.

Dependencies flow downward. The EmbeddingEngine supplies features, recognition, and detector,
features supplies fifo and recognition, and detector composes the two. L1 and L2 never depend on
each other."""
