"""Speaker-to-listener appraisal: two reaction tables plus SOUL-weighted redistribution.

Variant C from docs/MoodAppraisalExperiment.md, split into two tables. GoEmotions
classifies what a speaker expressed; a listener with their own identity rarely feels the
same emotion back. For each emotion the speaker expressed, the tables list the candidates
the listener might feel, and _appraise spreads the source mass across those candidates
weighted by the listener's SOUL profile.

The tables are static and hand-crafted, so they miss nuance, but they give a useful
baseline signal.
"""

import re
from .labels import GO_EMOTIONS_LABELS, LABEL_INDEX


# Lightweight text heuristic. Speech acts and second-person pronouns mean the speaker is
# acting on Eva; "I/we/me/us" means they're expressing about themselves (or a group/3rd
# party); neither means pure contagion (the caller skips appraisal). Crude, so replace it
# with structured speaker metadata from AudioSense when that exists.
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
_SENSE_TAG_RE = re.compile(r"^(?:I heard|[\w\s]+ said):\s*", re.I)


def strip_sense_tag(text: str) -> str:
    """Drop the senses-layer source prefix ('Adam said:', 'I heard:')."""
    return _SENSE_TAG_RE.sub("", text)


def detect_direction(text: str) -> str:
    """Classify 'directed', 'empathic', or 'contagion' from text cues.

    Expects the sense tag already stripped (see strip_sense_tag).
    """
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



def _appraise(
    raw: list[float],
    table: dict[str, set[str]],
    soul: list[float],
) -> list[float]:
    """Redistribute each source emotion across its candidate cluster.

    For the source emotion at index i with mass raw[i], spread that mass across
    table[label_i]'s candidates weighted by soul. If SOUL has no signal across the
    cluster, that source emotion doesn't contribute to the appraised mood.
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
