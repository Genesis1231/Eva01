"""EVA's speech channel — the non-visual half of a moment.

Mirrors _vision, but L1-only: the moment store (eva/core/moment.py) is the shared, cross-modal
recognition layer, so the speech channel only needs the cheap *trigger* (recent novelty) and the
agent-state text it embeds.

  window    — SpeechWindow: the rolling recent-speech buffer -> the LLM-free agent-state text.
  detector  — SpeechDetector: embeds the agent-state text, scores L1 recent-novelty (z + floor)
              -> a candidate trigger + the txt_key a moment is cued by.

Recognition ("have I heard this before?") is MomentDB.recall, not a per-channel L2 bank."""
