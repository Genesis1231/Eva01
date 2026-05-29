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

from eva.subconscious._mood.labels import GO_EMOTIONS_LABELS, LABEL_INDEX


# Lightweight text heuristic. Speech acts and second-person pronouns mean
# the speaker is acting on Eva; "I/we/me/us" means the speaker is expressing
# about self (or a group/3rd party); neither means pure contagion (caller
# skips appraisal). Crude — replace with structured speaker metadata from
# AudioSense when available (see Experiment doc §8).
_YOU_RE = re.compile(r"\byou(?:'?(?:re|ve|ll|d|rs))?\b|\byour\b", re.I)
_SELF_RE = re.compile(
    r"\bI(?:'?(?:m|ve|ll|d))?\b|\b(?:we|us|our|ours|me|my|mine)\b", re.I,
)
_THANKS_RE = re.compile(
    r"\b(?:thanks|thank\s+you|thx|appreciate\s+(?:it|that|this|everything))\b",
    re.I,
)
_IMPERATIVE_RE = re.compile(
    r"^\s*(?:please\s+)?(?:"
    r"look\s+(?:at|up|over)\b|listen\s+to\b|come\s+(?:here|over|back|on)\b|"
    r"help(?:\s+(?:me|us|with|out)\b|[.!?]*$)|"
    r"(?:tell|show|wait|stop|start|remember|forget|explain|check|open|"
    r"close|read|write|say|speak|watch|search|find|think|try|give|take|"
    r"send|call|wake)\b"
    r")",
    re.I,
)

_INTERJECTION_RE = re.compile(
    r"\boh+\s+my(\s+(?:god|goodness|gosh|lord|word))?\b", re.I,
)
_SECOND_PERSON_DISCOURSE_RE = re.compile(
    r"^\s*you\s+(?:know(?:\s+what)?|see)[,;:!?]+\s*",
    re.I,
)


def detect_direction(text: str) -> str:
    """Return 'directed', 'empathic', or 'contagion' from text cues."""
    
    _, separator, utterance = text.partition(":")
    if separator:
        # strip "Adam said:" 
        text = utterance.strip()
    text = _SECOND_PERSON_DISCOURSE_RE.sub("", text)
    if (
        _YOU_RE.search(text)
        or _THANKS_RE.search(text)
        or _IMPERATIVE_RE.search(text)
    ):
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
# includes embarrassment reflex (Brown & Levinson
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
    If SOUL has no signal across the cluster, the source emotion does
    not contribute to the appraised mood.
    """
    out = [0.0] * len(raw)
    for i, p in enumerate(raw):
        candidates = table.get(GO_EMOTIONS_LABELS[i], {GO_EMOTIONS_LABELS[i]})
        weights = [soul[LABEL_INDEX[c]] for c in candidates]
        total = sum(weights)
        if total <= 0:
            continue
        for c, w in zip(candidates, weights):
            out[LABEL_INDEX[c]] += p * (w / total)

    return [min(1.0, v) for v in out]
