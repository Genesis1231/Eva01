"""
EVA's mind.

Three concurrent components sharing two buffers:
    Senses  →  SenseBuffer  →  Brain  →  ActionBuffer  →  Actions
"""

import asyncio

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from config import logger, eva_configuration, DATA_DIR, Config
from eva.agent.chatagent import ChatAgent
from eva.core.graph import Brain
from eva.core.memory import MemoryDB
from eva.senses import SenseBuffer, AudioSense, CameraSense
from eva.actions import ActionBuffer, VoiceActor, Screen
from eva.senses.audio.transcriber import Transcriber
from eva.senses.vision.describer import Describer
from eva.senses.vision.identifier import Identifier
from eva.actions.voice.speaker import Speaker
from eva.core.people import PeopleDB
from eva.core.journal import JournalDB
from eva.core.db import SQLiteHandler


async def weave(
    config: Config, 
    db: SQLiteHandler, 
    checkpointer: AsyncSqliteSaver | None = None
):
    """Wire up senses, brain, and actions. Return shared buffers and components."""

    logger.debug("Assembling EVA's core components...")
    loop = asyncio.get_running_loop()

    # Shared buffers
    action_buffer = ActionBuffer()
    sense_buffer = SenseBuffer()
    sense_buffer.attach_loop(loop)

    # People & Memory
    people_db = PeopleDB(db)
    journal_db = JournalDB(db)
    
    await asyncio.gather(
        people_db.init_db(),
        journal_db.init_db(),
    )
    memory_db = MemoryDB(config.UTILITY_MODEL, people_db, journal_db)
    
    # initialize vision sense
    describer = Describer(config.VISION_MODEL)
    identifier = Identifier(people_db)
    camera_sense = CameraSense(describer, identifier=identifier, source=config.CAMERA_URL)
    if camera_sense.is_available:
        camera_sense.start(sense_buffer)

    # Actions — register handlers on the shared buffer
    speaker = Speaker(config.TTS_MODEL, config.LANGUAGE)
    voice_actor = VoiceActor(speaker)
    voice_actor.register(action_buffer)

    screen = Screen()
    screen.register(action_buffer)

    # initialize transcriber and audio sense
    transcriber = Transcriber(config.STT_MODEL)
    audio_sense = AudioSense(transcriber, keyboard=True, voice_actor=voice_actor)
    audio_sense.start(sense_buffer)

    # Brain — ChatAgent owns LLM + tools, Brain owns workflow + memory
    agent = ChatAgent(config.CHAT_MODEL, action_buffer, people_db=people_db)
    brain = Brain(agent, memory=memory_db, checkpointer=checkpointer)

    return sense_buffer, action_buffer, audio_sense, camera_sense, voice_actor, brain


async def breathe(sense_buffer: SenseBuffer, brain: Brain) -> None:
    """The conscious loop — EVA's mind."""

    while True:
        entry = await sense_buffer.get()

        try:
            await brain.invoke(entry)
        except Exception as e:
            logger.error(f"EVA: brain error — {e}")

async def wake() -> None:
    """Launch EVA — senses, mind, and voice running concurrently."""
    load_dotenv()

    # Ensure DB directory exists
    graph_db = DATA_DIR / "database" / "eva_graph.db"
    graph_db.parent.mkdir(parents=True, exist_ok=True)

    logger.debug("EVA is waking up...")

    db = SQLiteHandler()

    async with AsyncSqliteSaver.from_conn_string(str(graph_db)) as checkpointer:
        sense_buffer, action_buffer, audio_sense, camera_sense, voice_actor, brain = await weave(eva_configuration, db, checkpointer)

        logger.debug(f"EVA: session {brain.thread_id} — ready.")
        await action_buffer.put("speak", "I am ready!")
        print("\n   ... PRESS SPACE to talk to EVA ...")

        try:
            await asyncio.gather(
                breathe(sense_buffer, brain),
                action_buffer.start_loop(),
            )
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            # Journal
            messages = await brain.get_messages()
            if messages:
                try:
                    await brain.memory.flush(messages, session_id=brain.thread_id)
                except Exception as e:
                    logger.error(f"EVA: failed to flush memory — {e}")

            audio_sense.stop()
            if camera_sense:
                await camera_sense.stop()
            await voice_actor.stop()
            await db.close_all()
            logger.debug("EVA is falling asleep... Bye!")


if __name__ == "__main__":
    asyncio.run(wake())
