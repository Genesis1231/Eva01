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
