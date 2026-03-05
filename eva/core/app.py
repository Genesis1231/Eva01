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
from eva.actions import ActionBuffer, VoiceActor
from eva.senses.audio.transcriber import Transcriber
from eva.senses.vision.describer import Describer
from eva.senses.vision.identifier import Identifier
from eva.actions.voice.speaker import Speaker
from eva.core.people import PeopleDB


async def weave(config: Config, checkpointer: AsyncSqliteSaver = None):
    """Wire up senses, brain, and actions. Return shared buffers and components."""

    logger.debug("Weaving EVA's core components...")
    loop = asyncio.get_running_loop()

    # Shared buffers
    action_buffer = ActionBuffer()
    sense_buffer = SenseBuffer()
    sense_buffer.attach_loop(loop)

    # People & Memory
    people_db = PeopleDB()
    memory_db = MemoryDB(config.UTILITY_MODEL, people_db=people_db)

    # initialize vision sense
    describer = Describer(config.VISION_MODEL)
    identifier = Identifier(people_db)
    camera_sense = CameraSense(describer, identifier=identifier, source=config.CAMERA_URL)
    if camera_sense:
        camera_sense.start(sense_buffer)

    # Actions — create before AudioSense so interrupt-on-speak works
    speaker = Speaker(config.TTS_MODEL, config.LANGUAGE)
    voice_actor = VoiceActor(action_buffer, speaker)

    # initialize transcriber and audio sense
    transcriber = Transcriber(config.STT_MODEL)
    audio_sense = AudioSense(transcriber, keyboard=True, voice_actor=voice_actor)
    audio_sense.start(sense_buffer)

    # Brain — ChatAgent owns LLM + tools, Brain owns workflow
    agent = ChatAgent(config.CHAT_MODEL, action_buffer, memory=memory_db)
    brain = Brain(agent, checkpointer=checkpointer)

    return sense_buffer, action_buffer, audio_sense, camera_sense, voice_actor, brain, memory_db


async def breathe(sense_buffer: SenseBuffer, brain: Brain) -> None:
    """The conscious loop — EVA's mind."""

    while True:
        entry = await sense_buffer.get()
        logger.debug(f"EVA: sensed [{entry.type}] — {entry.content[:60]}")

        try:
            sense = ("I hear: " if entry.type == "audio" else "I see: ") + entry.content
            print(f"   [DBG] brain invoked with: {sense[:60]}")
            await brain.invoke(sense)
            print("   [DBG] brain done")

        except Exception as e:
            logger.error(f"EVA: brain error — {e}")

        await asyncio.sleep(0.1)


async def wake() -> None:
    """Launch EVA — senses, mind, and voice running concurrently."""
    load_dotenv()

    # Ensure DB directory exists
    graph_db = DATA_DIR / "database" / "eva_graph.db"
    graph_db.parent.mkdir(parents=True, exist_ok=True)

    logger.debug("EVA is waking up...")

    async with AsyncSqliteSaver.from_conn_string(str(graph_db)) as checkpointer:
        sense_buffer, action_buffer, audio_sense, camera_sense, voice_actor, brain, memory_db = await weave(eva_configuration, checkpointer)

        logger.debug(f"EVA: session {brain.thread_id} — ready.")
        await action_buffer.put("speak", "I am ready!.")
        print("\n   ... PRESS SPACE to talk to EVA ...")

        try:
            await asyncio.gather(
                breathe(sense_buffer, brain),
                voice_actor.start_loop(),
            )
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            # Journal
            messages = await brain.get_messages()
            if messages:
                await memory_db.flush(messages, session_id=brain.thread_id)

            audio_sense.stop()
            if camera_sense:
                await camera_sense.stop()
            await voice_actor.stop()
            logger.debug("EVA is falling asleep... Bye!")


if __name__ == "__main__":
    asyncio.run(wake())
