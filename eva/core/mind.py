"""
eva/core/mind.py — EVA's mind.

Three concurrent components sharing two buffers:
    Senses  →  SenseBuffer  →  Brain  →  ActionBuffer  →  Actions
"""

import asyncio
from dotenv import load_dotenv

from config import logger, eva_configuration
from eva.agent import ChatAgent
from eva.senses.sense_buffer import SenseBuffer
from eva.senses.audio.audio_sense import AudioSense
from eva.senses.audio.transcriber import Transcriber
from eva.actions.buffer import ActionBuffer
from eva.actions.voice.voice_actor import VoiceActor
from eva.actions.voice.speaker import Speaker


async def weave(config: dict) -> tuple[SenseBuffer, ActionBuffer, AudioSense, VoiceActor, ChatAgent]:
    """Wire up senses, brain, and actions. Return shared buffers and components."""
    
    logger.debug("Weaving EVA's core components...")
    
    loop = asyncio.get_running_loop()
        
    # Shared buffers
    action_buffer = ActionBuffer()
    sense_buffer = SenseBuffer()
    sense_buffer.attach_loop(loop)

    # Senses
    transcriber = Transcriber(config["STT_MODEL"], config["LANGUAGE"])
    audio_sense = AudioSense(transcriber, keyboard=True)
    audio_sense.start(sense_buffer)

    # Brain
    agent = ChatAgent(config["CHAT_MODEL"])

    # Actions
    speaker = Speaker(config["TTS_MODEL"], config["LANGUAGE"])
    voice_actor = VoiceActor(action_buffer, speaker)

    return sense_buffer, action_buffer, audio_sense, voice_actor, agent


async def breathe(sense_buffer: SenseBuffer, action_buffer: ActionBuffer, agent: ChatAgent) -> None:
    """The conscious loop — EVA's mind."""

    while True:
        entry = await sense_buffer.get()
        logger.debug(f"EVA: sensed [{entry.type}] — {entry.content[:60]}")

        try:
            sense = ("I heard: " if entry.type == "audio" else "I see: ") + entry.content
            response = await agent.arespond(sense=sense)
            text = response.get("response", "")
            
            if text:
                await action_buffer.put("speak", text)

        except Exception as e:
            logger.error(f"EVA: brain error — {e}")
        
        await asyncio.sleep(0.1)


async def wake() -> None:
    """Launch EVA — senses, mind, and voice running concurrently."""
    load_dotenv()
    config = eva_configuration

    logger.debug("EVA is waking up...")
    sense_buffer, action_buffer, audio_sense, voice_actor, agent = await weave(config)

    try:
        await asyncio.gather(
            breathe(sense_buffer, action_buffer, agent),
            voice_actor.start_loop(),
        )
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        audio_sense.stop()
        await voice_actor.stop()
        logger.debug("EVA is falling asleep... Bye!")


if __name__ == "__main__":
    asyncio.run(wake())
