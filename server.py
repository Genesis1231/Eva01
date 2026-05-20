"""
EVA — server mode.

WebSocket gateway: frontend senses → SenseBuffer → Brain → ActionBuffer → WebSocket.
Same pipeline as local mode, but WebSocket replaces mic/camera/speaker.
"""

import asyncio
import base64
import json
from contextlib import asynccontextmanager
from io import BytesIO

import numpy as np
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydub import AudioSegment

from config import logger, eva_configuration as config, DATA_DIR
from eva.core.app import breathe
from eva.core.graph import Brain
from eva.core.memory import MemoryDB
from eva.core.people import PeopleDB
from eva.core.journal import JournalDB
from eva.core.tasks import TaskDB
from eva.core.heart import Heart
from eva.database.db import SQLiteHandler
from eva.database.embeddings import EmbeddingEngine
from eva.database.vector_index import VectorIndex
from eva.senses.sense_buffer import SenseBuffer
from eva.senses.audio.transcriber import Transcriber
from eva.senses.vision.describer import Describer
from eva.actions.action_buffer import ActionBuffer, ActionEvent
from eva.actions.base import BaseAction
from eva.actions.voice.speaker import Speaker


# ── Constants ────────────────────────────────────────────────────────

MEDIA_ROOT = DATA_DIR / "media"
VALID_FILE_TYPES = {"images", "audio"}
MEDIA_TYPES = {"audio": "audio/mpeg", "images": "image/jpeg"}


# ── WebSocketActor ───────────────────────────────────────────────────

class WebSocketActor(BaseAction):
    """Bridges ActionBuffer → WebSocket.  Replaces VoiceActor for remote."""

    def __init__(self, speaker: Speaker, websocket: WebSocket, session_id: str):
        self.speaker = speaker
        self.websocket = websocket
        self.session_id = session_id
        self._active = True
        self._buffer: ActionBuffer | None = None

    def register(self, buffer: ActionBuffer) -> None:
        self._buffer = buffer
        buffer.on("speak", self._handle_speak)

    def deactivate(self) -> None:
        self._active = False
        if self._buffer is not None:
            self._buffer.off("speak", self._handle_speak)
            self._buffer = None

    async def _handle_speak(self, event: ActionEvent) -> None:
        if not self._active or not event.content:
            return
        try:
            audio_path = await self.speaker.get_audio(event.content)
            if not audio_path:
                return
            await self.websocket.send_json({
                "type": "audio",
                "content": audio_path,
                "text": event.content,
                "session_id": self.session_id,
            })
        except Exception:
            self._active = False

    async def stop(self) -> None:
        self._active = False


# ── Audio helper ─────────────────────────────────────────────────────

async def _transcribe_audio(audio_b64: str, transcriber: Transcriber) -> str | None:
    """Decode base64 MP3 from frontend and transcribe to text."""
    mp3_bytes = base64.b64decode(audio_b64)
    segment = AudioSegment.from_mp3(BytesIO(mp3_bytes))
    segment = segment.set_channels(1).set_frame_rate(16000)
    samples = np.array(segment.get_array_of_samples(), dtype=np.float32) / 32768.0

    result = await asyncio.to_thread(transcriber.transcribe, samples)
    if result:
        text, _ = result
        return text
    return None


# ── Server assembly ──────────────────────────────────────────────────

async def assemble_server(db: SQLiteHandler, checkpointer: AsyncSqliteSaver):
    """Wire up the core pipeline for server mode (no local senses/actions)."""
    loop = asyncio.get_running_loop()

    action_buffer = ActionBuffer()
    sense_buffer = SenseBuffer()
    sense_buffer.attach_loop(loop)

    # Semantic embedding engine
    embedder = EmbeddingEngine(config.EMBEDDING_MODEL, base_url=config.BASE_URL)

    # DBs
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

    # Task tool init
    from eva.tools import tasks as task_tools
    task_tools.init(task_db)

    # Models
    speaker = Speaker(config.TTS_MODEL, config.LANGUAGE)
    transcriber = Transcriber(config.STT_MODEL)
    describer = Describer(config.VISION_MODEL)
    await asyncio.gather(
        asyncio.to_thread(embedder.init_model),
        asyncio.to_thread(speaker.init_model),
        asyncio.to_thread(transcriber.init_model),
    )

    # Brain
    brain = Brain(
        model_name=config.CHAT_MODEL,
        action_buffer=action_buffer,
        people=people_db.get_all(),
        memory=memory_db,
        checkpointer=checkpointer,
    )

    # Heart
    heart = Heart(sense_buffer, task_db, config.HEARTBEAT_INTERVAL, is_busy=brain.is_busy)

    return sense_buffer, action_buffer, brain, heart, speaker, transcriber, describer


# ── Lifespan ─────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()

    graph_db = DATA_DIR / "database" / "eva_graph.db"
    graph_db.parent.mkdir(parents=True, exist_ok=True)
    eva_db = SQLiteHandler()

    async with AsyncSqliteSaver.from_conn_string(str(graph_db)) as checkpointer:
        sense_buffer, action_buffer, brain, heart, speaker, transcriber, describer = (
            await assemble_server(eva_db, checkpointer)
        )

        app.state.sense_buffer = sense_buffer
        app.state.action_buffer = action_buffer
        app.state.brain = brain
        app.state.speaker = speaker
        app.state.transcriber = transcriber
        app.state.describer = describer

        tasks = [
            asyncio.create_task(heart.start()),
            asyncio.create_task(breathe(sense_buffer, brain)),
            asyncio.create_task(action_buffer.start_loop()),
        ]

        logger.info("EVA server is awake.")
        yield

        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

        await brain.shutdown()
        await action_buffer.stop()
        await eva_db.close_all()
        logger.info("EVA server has gone to sleep.")


# ── FastAPI app ──────────────────────────────────────────────────────

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ───────────────────────────────────────────────────────────

@app.get("/download/{file_type}/{filename}")
async def download_file(file_type: str, filename: str, session_id: str | None = None):
    """Serve audio and image files to the frontend."""
    if file_type not in VALID_FILE_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid file type: {file_type}.")

    file_path = MEDIA_ROOT / file_type / filename
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    response = FileResponse(file_path, filename=filename, media_type=MEDIA_TYPES[file_type])
    if session_id:
        response.headers["Cache-Control"] = "private, max-age=60, must-revalidate"
        response.headers["Vary"] = "session_id"
    else:
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    return response


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()

    state = app.state
    session_id = f"session_{client_id}"

    await websocket.send_json({"type": "receive_start", "session_id": session_id})

    # Create actor for this connection
    ws_actor = WebSocketActor(state.speaker, websocket, session_id)
    ws_actor.register(state.action_buffer)

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type")
            content = msg.get("content", "")

            if msg_type == "text" and content:
                state.sense_buffer.push("speech", f"I heard: {content}")

            elif msg_type == "audio" and content:
                text = await _transcribe_audio(content, state.transcriber)
                if text:
                    state.sense_buffer.push("speech", f"I heard: {text}")

            elif msg_type in ("frontImage", "backImage") and content:
                description = await state.describer.describe(content)
                if description:
                    state.sense_buffer.push("observation", f"I see: {description}")

            elif msg_type == "over":
                pass  # end-of-batch signal

    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
    finally:
        ws_actor.deactivate()


# ── Entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8080)
