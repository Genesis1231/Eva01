"""
VoiceActor:
    EVA's voice — owns its own speech queue with a worker task so the
    global action bus never blocks on speech timing.

    register(buffer) -> registers speak/interrupt handlers on ActionBuffer
    start()          -> launches the worker task
    interrupt()      -> async: signals in-flight speech to stop and drops queued speech
    interrupt_from_thread() -> thread-safe barge-in for AudioSense
    stop()           -> stops worker and releases speaker
"""

import asyncio
import time
from typing import Callable, Optional

from config import logger

from .speaker import Speaker
from ..action_buffer import ActionBuffer, ActionEvent
from ..base import BaseAction


class VoiceActor(BaseAction):
    """
    Speech runs serially through an internal worker; the global ActionBuffer
    dispatcher only ever enqueues, so an interrupt event behind a speak event
    can preempt instead of waiting.
    """

    # Politeness grace: how long the floor must stay clear before EVA starts talking.
    # Deliberately short — the mic's VAD hangover (MIN_SILENCE, 0.8s) already keeps the
    # floor held through breath pauses, so the felt turn gap is the sum (~1s). Tune the
    # two together, not separately.
    _GRACE = 0.2

    def __init__(
        self,
        speaker: Speaker,
        is_user_speaking: Optional[Callable[[], bool]] = None,
    ):
        self.speaker = speaker
        self.is_user_speaking = is_user_speaking   # the floor signal; None = no gating
        self._queue: asyncio.Queue[ActionEvent] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None
        self._current_speech: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def is_speaking(self) -> bool:
        return self._current_speech is not None

    @property
    def is_playing(self) -> bool:
        """True while her voice is physically in the air (synthesis excluded).
        The mic's echo handling keys on this — synthesis time is clean air."""
        return self.speaker.is_playing

    @property
    def playback_level(self) -> float:
        """Instantaneous output RMS — the mic compares what it hears against this."""
        return self.speaker.playback_level

    def register(self, buffer: ActionBuffer) -> None:
        buffer.on("speak", self._enqueue)
        buffer.on("interrupt", self._handle_interrupt)

    async def _enqueue(self, event: ActionEvent) -> None:
        if event.content:
            await self._queue.put(event)

    async def _handle_interrupt(self, event: ActionEvent) -> None:
        await self.interrupt()

    async def interrupt(self) -> None:
        """Signal stop and drop queued speech.

        Does NOT cancel the in-flight asyncio task — Python threads can't
        be killed, so cancelling would lie about state while the thread
        keeps running (risk: overlap, audio after interrupt, use-after-close
        on shutdown). The worker's await on _current_speech blocks until
        the thread actually returns; is_speaking stays True until then.
        """
        self.speaker.stop_speaking()

        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        logger.debug("VoiceActor: stop signalled, queue drained.")

    def interrupt_from_thread(self) -> None:
        """Thread-safe barge-in: kill audio now, schedule full cleanup."""
        self.speaker.stop_speaking()
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self.interrupt(), self._loop)

    async def _await_floor(self) -> None:
        """Politeness: hold the next utterance until the user's floor has been clear
        for _GRACE. The user always wins the floor — EVA waits, she never talks over."""
        if self.is_user_speaking is None:
            return
        clear_since: float | None = None
        while True:
            if self.is_user_speaking():
                clear_since = None
            else:
                now = time.monotonic()
                if clear_since is None:
                    clear_since = now
                elif now - clear_since >= self._GRACE:
                    return
            await asyncio.sleep(0.1)

    async def _worker(self) -> None:
        """Consume the speech queue serially. Waits for each thread to finish
        before pulling the next event, so interrupted speech can't overlap
        with a follow-up speak."""
        while True:
            event = await self._queue.get()
            if not event.content:
                continue

            language = (event.metadata or {}).get("language", "en")

            # Synthesize FIRST — it's silent, so it conflicts with no one — and
            # buffer the audio. Only playback is gated on the floor: when the
            # user finishes, she answers instantly instead of paying synthesis
            # latency on top of the wait. A barge-in meanwhile is still honored:
            # the stop flag it sets makes play() skip the buffered utterance.
            try:
                prepared = await asyncio.to_thread(
                    self.speaker.synthesize, event.content, language
                )
            except Exception as e:
                logger.error(f"VoiceActor: synthesis error — {e}")
                prepared = None

            await self._await_floor()

            if prepared is not None:
                self._current_speech = asyncio.create_task(
                    asyncio.to_thread(self.speaker.play, prepared, event.content)
                )
            else:
                # Backend can't split (Edge/ElevenLabs): synthesize+play in one call.
                self._current_speech = asyncio.create_task(
                    asyncio.to_thread(self.speaker.speak, event.content, language)
                )
            try:
                await self._current_speech
            except Exception as e:
                logger.error(f"VoiceActor: speech error — {e}")
            finally:
                self._current_speech = None

    async def start(self) -> None:
        """Launch the speech worker."""
        self._loop = asyncio.get_running_loop()
        self._worker_task = asyncio.create_task(self._worker())

    async def stop(self) -> None:
        await self.interrupt()

        if self._current_speech and not self._current_speech.done():
            try:
                await self._current_speech
            except Exception as e:
                logger.error(f"VoiceActor: error during stop — {e}")

        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except (asyncio.CancelledError, Exception):
                pass

        self.speaker.close()
        logger.debug("Voice Actor stopped.")
