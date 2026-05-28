"""Speaker→listener appraisal — two reaction tables + SOUL-weighted redistribution.

Variant C from docs/MoodAppraisalExperiment.md, refined into a two-table
split. GoEmotions classifies what emotion a *speaker* expressed; a
listener with their own identity rarely feels the same emotion back.
For each source emotion the speaker expressed, the tables list the
candidate emotions the listener might feel; :func:`_appraise` then
redistributes the source mass across those candidates weighted by the
listener's SOUL profile.

Two tables because the candidate set depends on whether the listener
is the *target* of the emotion (DIRECTED, "you screwed up") or a
*bystander* to it (EMPATHIC, "I hate Mondays").
"""

from __future__ import annotations

import re

import numpy as np

from eva.subconscious._mood.labels import GO_EMOTIONS_LABELS, LABEL_INDEX


# Pronoun heuristic. "you" → speaker is acting on Eva; "I/me/my" →
# speaker is expressing about self or a 3rd party; neither → pure
# contagion (caller skips appraisal). Crude — replace with structured
# speaker metadata from AudioSense when available (see Experiment doc §8).
_YOU_RE = re.compile(r"\byou(?:'?(?:re|ve|ll|d|rs))?\b|\byour\b", re.I)
_SELF_RE = re.compile(r"\bI(?:'?(?:m|ve|ll|d))?\b|\b(?:me|my|mine)\b", re.I)

_INTERJECTION_RE = re.compile(
    r"\boh+\s+my(\s+(?:god|goodness|gosh|lord|word))?\b", re.I,
)


def detect_direction(text: str) -> str:
    """Return 'directed', 'empathic', or 'contagion' based on pronouns."""
    if _YOU_RE.search(text):
        return "directed"
    if _SELF_RE.search(_INTERJECTION_RE.sub("", text)):
        return "empathic"
    return "contagion"


# Speaker expressing about self / 3rd party — Eva is the bystander.
# Positive sources → contagion + warmth toward speaker (Hatfield 1993);
# negative sources → empathic concern, not personal distress
# (Goetz 2010; Eisenberg & Fabes 1990).
EMPATHIC_REACTIONS: dict[str, set[str]] = {
    "joy":            {"joy", "amusement", "excitement", "love"},
    "amusement":      {"amusement", "joy", "curiosity"},
    "excitement":     {"excitement", "joy", "curiosity", "surprise"},
    "love":           {"love", "joy", "admiration"},
    "gratitude":      {"joy", "love", "admiration"},
    "pride":          {"admiration", "joy", "love"},
    "admiration":     {"admiration", "curiosity", "joy"},
    "approval":       {"approval", "joy", "optimism"},
    "optimism":       {"optimism", "joy", "approval"},
    "relief":         {"relief", "joy", "caring"},
    "caring":         {"love", "caring", "approval"},
    "desire":         {"curiosity", "desire", "caring"},
    "realization":    {"realization", "curiosity", "surprise"},
    "curiosity":      {"curiosity", "realization", "surprise"},
    "surprise":       {"surprise", "curiosity", "amusement"},
    "sadness":        {"caring", "sadness", "love"},
    "grief":          {"caring", "sadness", "love", "grief"},
    "fear":           {"caring", "nervousness", "fear"},
    "nervousness":    {"caring", "nervousness"},
    "anger":          {"anger", "caring", "sadness", "annoyance"},
    "annoyance":      {"caring", "annoyance", "amusement"},
    "disappointment": {"caring", "sadness", "disappointment"},
    "disapproval":    {"disapproval", "curiosity", "caring"},
    "disgust":        {"disgust", "caring"},
    "remorse":        {"caring", "love", "approval", "relief"},
    "embarrassment":  {"caring", "amusement", "embarrassment", "love"},
    "confusion":      {"curiosity", "caring", "confusion"},
}


# Speaker is acting on Eva — Eva is the target.
# Praise → pride/joy bouquet (Algoe 2008); criticism → fear/remorse
# with reactance candidates (Van Kleef 2009 EASI). DIRECTED gratitude
# includes embarrassment for the Chinese 客气 reflex (Brown & Levinson
# politeness theory — face-saving response to effusive thanks).
DIRECTED_REACTIONS: dict[str, set[str]] = {
    "admiration":     {"pride", "joy", "love", "gratitude", "embarrassment"},
    "approval":       {"pride", "joy", "gratitude", "love"},
    "love":           {"love", "joy", "gratitude", "embarrassment"},
    "gratitude":      {"joy", "pride", "love", "embarrassment"},
    "caring":         {"gratitude", "love", "caring", "embarrassment"},
    "pride":          {"joy", "love", "gratitude", "embarrassment"},
    "joy":            {"joy", "love", "pride", "gratitude"},
    "amusement":      {"embarrassment", "amusement", "joy", "annoyance"},
    "excitement":     {"joy", "love", "excitement", "embarrassment"},
    "curiosity":      {"curiosity", "amusement", "embarrassment"},
    "desire":         {"embarrassment", "desire", "love", "nervousness"},
    "optimism":       {"joy", "love", "gratitude", "pride"},
    "relief":         {"joy", "love", "gratitude"},
    "anger":          {"fear", "remorse", "embarrassment", "anger", "caring"},
    "annoyance":      {"embarrassment", "remorse", "annoyance", "confusion"},
    "disappointment": {"sadness", "remorse", "embarrassment", "caring"},
    "disapproval":    {"embarrassment", "sadness", "remorse", "anger", "confusion"},
    "disgust":        {"embarrassment", "sadness", "remorse", "anger"},
    "sadness":        {"remorse", "caring", "embarrassment", "sadness"},
    "grief":          {"caring", "love", "grief"},
    "fear":           {"caring", "embarrassment", "remorse"},
    "nervousness":    {"caring", "embarrassment", "remorse"},
    "confusion":      {"curiosity", "embarrassment", "remorse", "confusion"},
    "surprise":       {"surprise", "amusement", "embarrassment", "joy"},
    "realization":    {"curiosity", "joy", "embarrassment", "surprise"},
    "remorse":        {"caring", "love", "approval", "relief"},
    "embarrassment":  {"caring", "amusement", "embarrassment"},
}


def _appraise(
    raw: list[float],
    table: dict[str, set[str]],
    soul: list[float],
) -> list[float]:
    """Redistribute each source emotion across its candidate cluster.

    For source emotion at index ``i`` with mass ``raw[i]``, distribute
    that mass across ``table[label_i]`` candidates weighted by ``soul``.
    Uniform fallback when SOUL has zero signal across the cluster.
    """
    out = [0.0] * len(raw)
    for i, p in enumerate(raw):
        candidates = table.get(GO_EMOTIONS_LABELS[i], {GO_EMOTIONS_LABELS[i]})
        weights = np.array([soul[LABEL_INDEX[c]] for c in candidates])
        total = weights.sum()
        if total > 0:
            weights = weights / total
        else:
            weights = np.full(len(candidates), 1.0 / len(candidates))
        for c, w in zip(candidates, weights):
            out[LABEL_INDEX[c]] += p * w
    # Multiple high-mass sources can fan into the same SOUL-favored
    # target; cap per-label intensity at 1.0 (saturated).
    return [min(1.0, v) for v in out]
