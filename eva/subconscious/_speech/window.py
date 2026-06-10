"""SpeechWindow: one rolling speech buffer, two views.

The SenseBuffer is drained by the brain, so the speech channel keeps its own recency ring.
  current()  the last WINDOW_SECONDS: the LLM-free agent-state text the detector scores
             (names stripped, role-marked, length-bounded, 'quiet' when idle).
  since(t)   back to RETENTION: a forming moment's full transcript.

One buffer, one pop horizon: both views read the same deque and only RETENTION evicts. current()
filters at 20s rather than popping, since popping there would strand the older fragments since()
still needs. Fed from any thread (AudioSense), read from the gate loop. deque append/popleft are
atomic on opposite ends so no lock is needed, and reads snapshot with list() before filtering."""

import re
import time
from collections import deque

# Strip the sense prefixes AudioSense adds ("Adam said: ", "I heard: "): embed content, not the tag.
_PREFIX = re.compile(r"^(?:I (?:heard|see|observe|noticed)|[\w\s]+ said):\s*", re.IGNORECASE)

WINDOW_SECONDS = 20.0     # current()'s reach: the agent-state text / detector cue view
RETENTION = 1200.0        # how long fragments live for since(); > Experience MAX_AGE (900) + pre-roll
MAX_CHARS = 400           # bound current(); token count is the modality-balance knob
SENTINEL = "quiet"        # idle: a constant string -> constant e_txt -> ~zero text-novelty


class SpeechWindow:
    """A rolling buffer of recent speech fragments -> the agent-state text + moment transcripts."""

    def __init__(
        self,
        window_seconds: float = WINDOW_SECONDS,
        max_chars: int = MAX_CHARS,
        retention: float = RETENTION,
    ):
        self._window_seconds = window_seconds
        self._max_chars = max_chars
        self._retention = retention
        self._frags: deque[tuple[float, str, str]] = deque()   # (monotonic_ts, role, text)

    def add(self, content: str, role: str = "heard") -> None:
        """Add a transcript fragment (prefix stripped to content). Safe to call from any thread."""
        text = _PREFIX.sub("", content).strip()
        if text:
            self._frags.append((time.monotonic(), role, text))

    def _evict(self, now: float) -> None:
        """The single pop horizon: drop fragments older than RETENTION. Both views call it. current()'s
        20s view filters and never pops, otherwise since()'s longer reach would lose its tail."""
        cutoff = now - self._retention
        while self._frags and self._frags[0][0] < cutoff:
            self._frags.popleft()

    def current(self) -> str:
        """The agent-state text: the last WINDOW_SECONDS of speech (content-only, role-marked),
        or 'quiet' when idle."""
        now = time.monotonic()
        self._evict(now)
        cutoff = now - self._window_seconds
        frags = [f for f in list(self._frags) if f[0] >= cutoff]   # snapshot, then filter (no pop)

        if not frags:
            return SENTINEL
        text = "\n".join(f"{role}: {said}" for _, role, said in frags)

        return text[-self._max_chars:] if len(text) > self._max_chars else text

    def since(self, t: float) -> list[tuple[float, str, str]]:
        """Every fragment at or after monotonic time t (bounded by RETENTION), a forming moment's
        transcript. Returns raw (ts, role, text) triples; the encoder formats them."""
        self._evict(time.monotonic())
        return [f for f in list(self._frags) if f[0] >= t]
