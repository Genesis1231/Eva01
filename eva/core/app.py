"""
EVA's mind.
"""

import asyncio
from dataclasses import dataclass

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from config import logger, eva_configuration, DATA_DIR, Config

from .graph import Brain
from .memory import MemoryDB
from .people import PeopleDB
from .moment import MomentDB
from .tasks import TaskDB
from .heart import Heart
from eva.subconscious.subconscious import Subconscious
from eva.subconscious._vision.detector import VisionDetector
from eva.subconscious._vision.recognition import RecognitionMemory
from eva.subconscious._speech.window import SpeechWindow
from eva.senses.sense_buffer import SenseBuffer
from eva.senses.audio.audio_sense import AudioSense
from eva.senses.audio.transcriber import Transcriber
from eva.senses.audio.speaker_identifier import SpeakerIdentifier
from eva.senses.vision.vision_sense import CameraSense
from eva.senses.vision.describer import Describer
from eva.senses.vision.face_identifier import FaceIdentifier

from eva.database.db import SQLiteHandler
from eva.database.embeddings import EmbeddingEngine
from eva.actions.action_buffer import ActionBuffer
from eva.actions.voice.voice_actor import VoiceActor
from eva.actions.machine.browser import Browser
from eva.actions.system import MotorSystem
from eva.actions.voice.speaker import Speaker

@dataclass(frozen=True, slots=True)
class Assembly:
    """Every component wake() needs to run and tear down. Returned by assemble()."""
    sense_buffer: SenseBuffer
    action_buffer: ActionBuffer
    brain: Brain
    heart: Heart
    motor_system: MotorSystem
    audio_sense: AudioSense
    embedder: EmbeddingEngine
    speech_window: SpeechWindow
    camera_sense: CameraSense | None = None
    subconscious: Subconscious | None = None

async def assemble(
    config: Config,
    db: SQLiteHandler,
    checkpointer: AsyncSqliteSaver | None = None,
) -> Assembly:
    """Wire up senses, brain, and actions. Return the full Assembly."""

    logger.debug("Assembling EVA's core components...")
    loop = asyncio.get_running_loop()

    # Shared buffers
    action_buffer = ActionBuffer()
    sense_buffer = SenseBuffer()
    sense_buffer.attach_loop(loop)

    # Semantic embedding engine (safe-degrading if provider/deps are unavailable)
    embedder = EmbeddingEngine(config.EMBEDDING_MODEL, base_url=config.EMBEDDING_URL)

    # People, Memory & Tasks — moments are the unified episodic + reflection store
    people_db = PeopleDB(db)
    moment_db = MomentDB(db)
    task_db = TaskDB(db)
    await asyncio.gather(
        people_db.init_db(),
        moment_db.init_db(),
        task_db.init_db(),
    )
    memory_db = MemoryDB(config.UTILITY_MODEL, people_db, moment_db, embedder)

    # Init task tool before Brain (Brain calls load_tools in __init__)
    from eva.tools import tasks as task_tools
    task_tools.init(task_db)
    
    # Initialize vision sense
    camera_sense = None
    face_identifier = None
    if config.CAMERA is not False:
        describer = Describer(config.VISION_MODEL)
        # face_identifier = FaceIdentifier(people_db) temply disable face ID.
        camera_sense = CameraSense(
            describer=describer,
            identifier=face_identifier,
            source=config.CAMERA
        )

        # Share the camera with the peak tool
        from eva.tools import peak as peak_tool
        peak_tool.init(camera_sense)

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
 
    # initialize transcriber and audio sense; the speech window taps every transcript
    # (the rolling recent-speech buffer behind the future speech channel / moment txt_key)
    speech_window = SpeechWindow()
    audio_sense = AudioSense(
        transcriber,
        speaker_identifier=speaker_identifier,
        on_interrupt=voice_actor.interrupt_from_thread,
        is_speaking=lambda: voice_actor.is_speaking,
        on_transcript=speech_window.add,
    )
    audio_sense.start(sense_buffer)

    # Brain — owns tools + workflow, Cortex owns LLM + prompt
    brain = Brain(
        model_name=config.MAIN_MODEL,
        action_buffer=action_buffer,
        memory=memory_db,
        people=people_db.get_all(),
        checkpointer=checkpointer,
    )

    # Heartbeat — autonomic vitals (storage / connectivity / embedding server)
    heart = Heart(db, config.HEARTBEAT_INTERVAL, embedding_url=config.EMBEDDING_URL)

    # Subconscious — the always-on visual-novelty gate (a concurrent peer of breathe).
    subconscious = None
    if camera_sense and camera_sense.is_available and embedder.enabled:
        session_dir = DATA_DIR / "session"
        session_dir.mkdir(parents=True, exist_ok=True)
        
        recognition_memory = await RecognitionMemory.seed(
            prior_stream = session_dir / "stream",
            cache_path = session_dir / "seed_embed.npy",
            engine = embedder,
        )

        vision_detector = VisionDetector(
            l2 = recognition_memory,
            engine = embedder
        )

        subconscious = Subconscious(
            sense_buffer,
            camera_sense,
            vision_detector,
            inspect_dir = session_dir / "inspects",
        )

    return Assembly(
        sense_buffer=sense_buffer,
        action_buffer=action_buffer,
        brain=brain,
        heart=heart,
        motor_system=motor_system,
        audio_sense=audio_sense,
        embedder=embedder,
        speech_window=speech_window,
        camera_sense=camera_sense,
        subconscious=subconscious,
    )


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
        eva = await assemble(
            config=eva_configuration,
            db=eva_db,
            checkpointer=checkpointer,
        )

        try:
            await eva.motor_system.start()
            loops = [
                eva.heart.start(),
                breathe(eva.sense_buffer, eva.brain),
                eva.action_buffer.start_loop(),
            ]

            if eva.subconscious is not None:
                loops.append(eva.subconscious.start())
                
            await asyncio.gather(*loops)
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            await eva.brain.shutdown()

            eva.audio_sense.stop()
            if eva.subconscious is not None:
                await eva.subconscious.stop()      # also releases the camera
            elif eva.camera_sense is not None:
                eva.camera_sense.release()

            await eva.embedder.aclose()
            await eva.motor_system.shutdown()
            await eva.action_buffer.stop()
            await eva_db.close_all()

if __name__ == "__main__":
    asyncio.run(wake())
