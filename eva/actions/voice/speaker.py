"""
Speaker factory: selects TTS model (edge/elevenlabs/kokoro) and delegates. 
  speak(answer, language) -> streams audio via model.eva_speak
  get_audio(text) -> writes mp3 to media folder, returns relative path
  stop_speaking() -> delegates to model.stop_playback
"""

from datetime import datetime
from config import logger, DATA_DIR
from typing import Callable

class Speaker:
    """
    The Speaker class is responsible for initializing and managing the text-to-speech models.
    It provides methods to create different speaker models and speak the given text.

    Attributes:
        model_selection (str): The selected speaker model name.
        model_factory (dict): A dictionary mapping model names to their corresponding creation methods.
    """
    
    def __init__(self, speaker_model: str = "kokoro", language: str = "en")-> None:
        self._model_name: str = speaker_model.upper()
        self._language: str = language
        self.model = None
    
    def _get_model_factory(self) -> dict[str, Callable]:
        return {
            "EDGE"       : self._create_edge_model,
            "ELEVENLABS" : self._create_elevenlab_model,
            "KOKORO"     : self._create_kokoro_model,
        }

    def _create_edge_model(self):
        from .model_edge import EdgeSpeaker

        try:
            return EdgeSpeaker()
        except Exception as e:
            raise Exception(f"Error: Failed to initialize Edge TTS model {str(e)}")

    def _create_elevenlab_model(self):
        from .model_elevenlabs import ElevenLabsSpeaker
        
        try:
            return ElevenLabsSpeaker()
        except Exception as e:
            raise Exception(f"Error: Failed to initialize ElevenLabs model {str(e)} ")


    def _create_kokoro_model(self):
        from .model_kokoro import KokoroSpeaker
        
        try:
            return KokoroSpeaker()
        except Exception as e:
            raise Exception(f"Error: Failed to initialize Kokoro model {str(e)}")
    
    def init_model(self):
        """Initialize the selected TTS model once (safe to call multiple times)."""
        if self.model is not None:
            return self.model

        model_factory = self._get_model_factory()
        model = model_factory.get(self._model_name)
        if model is None:
            raise ValueError(f"Error: Model {self._model_name} is not supported.")

        self.model = model()
        logger.debug(f"Speaker: {self._model_name} is ready.")
        return self.model

    def stop_speaking(self) -> None:
        """ Stop the speaker model. Thread-safe. """
        if self.model is None or not hasattr(self.model, 'stop_playback'):
            logger.error("Speaker: TTS model not initialized or does not support stopping playback.")
            return

        self.model.stop_playback()
        
    def speak(self, answer: str, language: str = "en"):
        """ Speak the given text. Blocking — run via to_thread. """

        if self.model is None or not hasattr(self.model, 'eva_speak'):
            logger.error("Speaker: TTS model not initialized or does not support speaking.")
            raise
        
        try:
            print(f"\n({datetime.now().strftime('%H:%M:%S')}) EVA: {answer}")
            self.model.eva_speak(answer, language)

        except Exception as e:
            raise Exception(f"Error: Failed to speak {str(e)} ")
        
    async def get_audio(self, text: str) -> str:
        """ Generate audio from text and save it to the media folder """
        if self.model is None or not hasattr(self.model, 'generate_audio'):
            logger.error("Speaker: TTS model not initialized or does not support audio generation.")
            return ""
        
        return await self.model.generate_audio(text, self._language, DATA_DIR / 'media')

    def close(self) -> None:
        """Release the underlying TTS model resources."""
        if self.model is not None and hasattr(self.model, 'close'):
            self.model.close()
            logger.debug(f"Speaker: {self._model_name} Model released.")