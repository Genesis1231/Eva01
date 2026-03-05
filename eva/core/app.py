"""
EVA's mind.

Three concurrent components sharing two buffers:
    Senses  →  SenseBuffer  →  Brain  →  ActionBuffer  →  Actions
"""

import asyncio
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from config import logger, eva_configuration
from config.config import Config
from eva.agent.chatagent import ChatAgent
from eva.core.graph import Brain, THREAD_ID
from eva.core.memory import MemoryDB
from eva.senses import SenseBuffer, AudioSense, CameraSense
from eva.actions import ActionBuffer, VoiceActor
from eva.senses.audio.transcriber import Transcriber
from eva.senses.vision.describer import Describer
from eva.senses.vision.identifier import Identifier
from eva.actions.voice.speaker import Speaker
from eva.core.people import PeopleDB

DB_PATH = "data/database/eva_graph.db"


async def weave(config: Config, checkpointer=None):
    """Wire up senses, brain, and actions. Return shared buffers and components."""

    logger.debug("Weaving EVA's core components...")
    loop = asyncio.get_running_loop()

    # Shared buffers
    action_buffer = ActionBuffer()
    sense_buffer = SenseBuffer()
    sense_buffer.attach_loop(loop)

    # Memory
    memory_db = MemoryDB()

    # initialize vision sense
    describer = Describer(config.VISION_MODEL)
    people_db = PeopleDB()
    identifier = Identifier(people_db)
    camera_sense = CameraSense(describer, identifier=identifier, source=config.CAMERA_URL)
    if camera_sense:
        camera_sense.start(sense_buffer)

    # initialize transcriber and audio sense
    transcriber = Transcriber(config.STT_MODEL)
    audio_sense = AudioSense(transcriber, keyboard=True)
    audio_sense.start(sense_buffer)

    # Brain — ChatAgent owns LLM + tools, Brain owns workflow
    agent = ChatAgent(config.CHAT_MODEL, action_buffer, memory=memory_db)
    brain = Brain(agent, checkpointer=checkpointer)

    # Actions
    speaker = Speaker(config.TTS_MODEL, config.LANGUAGE)
    voice_actor = VoiceActor(action_buffer, speaker)

    logger.debug("EVA: successfully initialized all components. Now ready to go!")
    return sense_buffer, action_buffer, audio_sense, camera_sense, voice_actor, brain, memory_db


async def breathe(sense_buffer: SenseBuffer, brain: Brain) -> None:
    """The conscious loop — EVA's mind."""

    while True:
        entry = await sense_buffer.get()
        logger.debug(f"EVA: sensed [{entry.type}] — {entry.content[:60]}")

        try:
            sense = ("I hear: " if entry.type == "audio" else "I see: ") + entry.content
            await brain.invoke(sense)

        except Exception as e:
            logger.error(f"EVA: brain error — {e}")

        await asyncio.sleep(0.1)


async def _recover_checkpointer(brain: Brain, memory_db: MemoryDB, checkpointer) -> None:
    """If residual messages exist from a crash, flush them to journal and clear."""
    messages = await brain.get_messages()
    if not messages:
        return

    logger.debug(f"MemoryDB: recovering {len(messages)} residual messages from checkpointer.")
    session_id = f"recovered-{uuid.uuid4().hex[:8]}"
    memory_db.flush(messages, session_id=session_id)
    await _clear_checkpointer(checkpointer)


async def _flush_and_clear(brain: Brain, memory_db: MemoryDB, checkpointer) -> None:
    """Flush current session to journal and clear checkpointer."""
    messages = await brain.get_messages()
    if not messages:
        logger.debug("MemoryDB: no messages to flush on shutdown.")
        return

    session_id = f"session-{uuid.uuid4().hex[:8]}"
    memory_db.flush(messages, session_id=session_id)
    await _clear_checkpointer(checkpointer)


async def _clear_checkpointer(checkpointer) -> None:
    """Clear all checkpointer data for our thread. AsyncSqliteSaver has no clear API."""
    try:
        async with checkpointer.conn.execute(
            "DELETE FROM checkpoints WHERE thread_id = ?", (THREAD_ID,)
        ) as _:
            pass
        async with checkpointer.conn.execute(
            "DELETE FROM writes WHERE thread_id = ?", (THREAD_ID,)
        ) as _:
            pass
        await checkpointer.conn.commit()
        logger.debug("MemoryDB: checkpointer cleared.")
    except Exception as e:
        logger.error(f"MemoryDB: failed to clear checkpointer — {e}")


async def wake() -> None:
    """Launch EVA — senses, mind, and voice running concurrently."""
    load_dotenv()
    config: Config = eva_configuration

    # Ensure DB directory exists
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)

    logger.debug("EVA is waking up...")

    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as checkpointer:
        sense_buffer, action_buffer, audio_sense, camera_sense, voice_actor, brain, memory_db = await weave(config, checkpointer)

        # Recover any residual messages from a previous crash
        await _recover_checkpointer(brain, memory_db, checkpointer)

        try:
            await asyncio.gather(
                breathe(sense_buffer, brain),
                voice_actor.start_loop(),
            )
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            # Flush session to journal before shutdown
            await _flush_and_clear(brain, memory_db, checkpointer)

            audio_sense.stop()
            if camera_sense:
                await camera_sense.stop()
            await voice_actor.stop()
            logger.debug("EVA is falling asleep... Bye!")


if __name__ == "__main__":
    asyncio.run(wake())
