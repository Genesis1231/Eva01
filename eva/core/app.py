"""
EVA's mind.

Three concurrent components sharing two buffers:
    Senses  →  SenseBuffer  →  Brain  →  ActionBuffer  →  Actions
"""

import asyncio
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from config import logger, eva_configuration, DATA_DIR, Config

from .graph import Brain
from .memory import MemoryDB
from .people import PeopleDB
from .journal import JournalDB
from .tasks import TaskDB
from .heart import Heart
from eva.senses.sense_buffer import SenseBuffer
from eva.senses.audio.audio_sense import AudioSense
from eva.senses.audio.transcriber import Transcriber
from eva.senses.audio.speaker_identifier import SpeakerIdentifier
from eva.senses.vision.vision_sense import CameraSense
from eva.senses.vision.describer import Describer
from eva.senses.vision.face_identifier import FaceIdentifier

from eva.database.db import SQLiteHandler
from eva.database.embeddings import EmbeddingEngine
from eva.database.vector_index import VectorIndex
from eva.actions.action_buffer import ActionBuffer
from eva.actions.voice.voice_actor import VoiceActor
from eva.actions.machine.browser import Browser
from eva.actions.system import MotorSystem
from eva.actions.voice.speaker import Speaker


async def assemble(
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

    # Semantic embedding engine (safe-degrading if provider/deps are unavailable)
    embedder = EmbeddingEngine(config.EMBEDDING_MODEL, base_url=config.BASE_URL)

    # People, Memory & Tasks
    people_db = PeopleDB(db)
    journal_vectors = VectorIndex(db, embedder, prefix="journal")
    journal_db = JournalDB(db, vectors=journal_vectors)
    task_db = TaskDB(db)
    await asyncio.gather(
        people_db.init_db(),
        journal_db.init_db(),
        task_db.init_db(),
    )
    memory_db = MemoryDB(config.UTILITY_MODEL, people_db, journal_db)

    # Init task tool before Brain (Brain calls load_tools in __init__)
    from eva.tools import tasks as task_tools
    task_tools.init(task_db)
    
    # Initialize vision sense
    camera_sense = None
    face_identifier = None
    if config.CAMERA is not False:
        describer = Describer(config.VISION_MODEL)
        face_identifier = FaceIdentifier(people_db)
        camera_sense = CameraSense(
            describer=describer, 
            identifier=face_identifier, 
            source=config.CAMERA
        )

    # Audio components
    speaker = Speaker(config.TTS_MODEL, config.LANGUAGE)
    transcriber = Transcriber(config.STT_MODEL)
    speaker_identifier = SpeakerIdentifier(people_db)

    # Initialize heavy models concurrently to reduce startup latency.
    init_tasks = [
        asyncio.to_thread(embedder.init_model),
        asyncio.to_thread(speaker.init_model),
        asyncio.to_thread(transcriber.init_model),
        asyncio.to_thread(speaker_identifier.init_model),
    ]
    if face_identifier is not None:
        init_tasks.append(asyncio.to_thread(face_identifier.init_model))
    await asyncio.gather(*init_tasks)
    
    # Actions — register handlers on the shared buffer
    voice_actor = VoiceActor(speaker)
    browser = Browser()
    motor_system = MotorSystem(
        action_buffer, 
        actions=[voice_actor, browser]
    )
 
    # initialize transcriber and audio sense

    audio_sense = AudioSense(
        transcriber,
        speaker_identifier=speaker_identifier,
        on_interrupt=voice_actor.interrupt_from_thread,
        is_speaking=lambda: voice_actor.is_speaking,
    )
    audio_sense.start(sense_buffer)

    if camera_sense is not None and camera_sense.is_available:
        camera_sense.start(sense_buffer)

    # Brain — owns tools + workflow, Cortex owns LLM + prompt
    brain = Brain(
        model_name=config.MAIN_MODEL,
        action_buffer=action_buffer,
        memory=memory_db,
        people=people_db.get_all(),
        checkpointer=checkpointer,
    )

    # Heartbeat
    heart = Heart(sense_buffer, task_db, config.HEARTBEAT_INTERVAL, is_busy=brain.is_busy)

    return sense_buffer, action_buffer, motor_system, audio_sense, camera_sense, brain, heart


async def breathe(sense_buffer: SenseBuffer, brain: Brain) -> None:
    """The conscious loop — EVA's mind."""

    while True:
        entry = await sense_buffer.get()

        try:
            await brain.invoke(entry)
        except Exception as e:
            logger.error(f"EVA: brain invoking error — {e}")

async def wake() -> None:
    """Launch EVA — senses, mind, and voice running concurrently."""
    logger.debug("Eva is waking up...")
    load_dotenv()

    # Ensure DB directory exists
    graph_db = DATA_DIR / "database" / "eva_graph.db"
    graph_db.parent.mkdir(parents=True, exist_ok=True)
    eva_db = SQLiteHandler()

    async with AsyncSqliteSaver.from_conn_string(str(graph_db)) as checkpointer:
        sense_buffer, action_buffer, motor_system, audio_sense, camera_sense, brain, heart = await assemble(
            config=eva_configuration, 
            db=eva_db, 
            checkpointer=checkpointer
        )

        try:
            await motor_system.start()
            await asyncio.gather(
                heart.start(),
                breathe(sense_buffer, brain),
                action_buffer.start_loop(),
            )
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            await brain.shutdown()  

            audio_sense.stop()
            if camera_sense is not None:
                await camera_sense.stop()
                
            await motor_system.shutdown()
            await action_buffer.stop()
            await eva_db.close_all()

if __name__ == "__main__":
    asyncio.run(wake())
