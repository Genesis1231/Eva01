"""GoEmotions label vocabulary — index alignment with model output.

The i-th probability emitted by SamLowe/roberta-base-go_emotions-onnx
corresponds to the i-th label here, so this list is also the canonical
index → name mapping for the mood vector.
"""

GO_EMOTIONS_LABELS: list[str] = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse",
    "sadness", "surprise", "neutral",
]

LABEL_INDEX: dict[str, int] = {
    name: i for i, name in enumerate(GO_EMOTIONS_LABELS)
}


# Speaker expressing about self / 3rd party — Eva is the bystander.
# Positive sources → contagion + warmth toward speaker;
# negative sources → empathic concern, not personal distress.
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
