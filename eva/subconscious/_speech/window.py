"""SpeechWindow — the rolling recent-speech buffer behind the agent-state text.

The SenseBuffer is a consumable queue (the brain drains it), so the speech channel keeps its own
short rolling window of recent transcripts — the text analog of L1's recency ring. `current()` turns
it into the LLM-free *agent-state text*: what's been said in the last WINDOW_SECONDS, content only
(speaker names stripped — read what was said, not the name tag), role-marked, length-bounded, with a
'quiet' sentinel when nothing's been said. Fed from any thread (AudioSense); read from the loop.
deque append/popleft are atomic, so no lock is needed."""

import re
import time
from collections import deque

# Strip the sense prefixes AudioSense adds ("Adam said: ", "I heard: ") — embed content, not the tag.
_PREFIX = re.compile(r"^(?:I (?:heard|see|observe|noticed)|[\w\s]+ said):\s*", re.IGNORECASE)

WINDOW_SECONDS = 20.0     # how far back the agent-state text reaches
MAX_CHARS = 400           # bound the text — token count is the modality-balance knob
SENTINEL = "quiet"        # idle: a constant string -> constant e_txt -> ~zero text-novelty


class SpeechWindow:
    """A rolling buffer of recent speech fragments -> the agent-state text."""

    def __init__(self, window_seconds: float = WINDOW_SECONDS, max_chars: int = MAX_CHARS):
        self._window_seconds = window_seconds
        self._max_chars = max_chars
        self._frags: deque[tuple[float, str, str]] = deque()   # (monotonic_ts, role, text)

    def add(self, content: str, role: str = "heard") -> None:
        """Add a transcript fragment (prefix stripped to content). Safe to call from any thread."""
        text = _PREFIX.sub("", content).strip()
        if text:
            self._frags.append((time.monotonic(), role, text))

    def current(self) -> str:
        """The agent-state text: recent speech (content-only, role-marked), or 'quiet' when idle."""
        cutoff = time.monotonic() - self._window_seconds
        while self._frags and self._frags[0][0] < cutoff:
            self._frags.popleft()
        frags = list(self._frags)                               # snapshot before joining
        if not frags:
            return SENTINEL
        text = "\n".join(f"{role}: {said}" for _, role, said in frags)
        return text[-self._max_chars:] if len(text) > self._max_chars else text
