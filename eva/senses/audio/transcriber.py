from typing import Dict, Optional, Callable, Tuple
from config import logger

class Transcriber:
    """
    The Transcriber class is responsible for transcribing audio clips using different models.
    
    Args:
        model_name (str): The name of the model to use for transcription. Default is "faster-whisper".
        language (str): The language code for transcription (e.g., "en").
    
    Examples:
        >>> transcriber = Transcriber(model_name="faster-whisper", language="en")
        >>> transcription = transcriber.transcribe(audioclip)
    """
    
    def __init__(self, model_name: str = "faster-whisper"):
        self._model_selection: str = model_name.upper()
        self.model = None
    
    def _get_model_factory(self) -> Dict[str, Callable]:
        return {
            "FASTER-WHISPER": self._create_fasterwhisper_model,
            "WHISPER": self._create_whisper_model,
        }
        
    def _create_fasterwhisper_model(self):
        from .models.model_fasterwhisper import FWTranscriber
        
        try:
            return FWTranscriber()
        except Exception as e:
            logger.error(f"Error: failed to load Faster Whisper Model: {e}")
            raise 
        
    def _create_whisper_model(self):
        from .models.model_whisper import WhisperTranscriber
        
        try:
            return WhisperTranscriber()
        except Exception as e:
            logger.error(f"Error: failed to load Whisper Model: {e}")
            raise 
        
    def init_model(self):
        """Initialize the selected transcription model once."""
        if self.model is not None:
            return self.model

        model_factory = self._get_model_factory()
        create_model = model_factory.get(self._model_selection)

        if create_model is None:
            raise ValueError(f"Error: Model {self._model_selection} is not supported")

        self.model = create_model()
        if hasattr(self.model, "init_model"):
            self.model.init_model()
        logger.debug(f"Transcriber: {self._model_selection} is ready.")
        return self.model


    def transcribe(self, audioclip) -> Optional[Tuple[str, str]]:
        """ Transcribe the given audio clip. """
        
        if self.model is None:
            logger.error("Transcriber: Model is not initialized. Call init_model() first.")
            return None
        
        try:
            transcription = self.model.transcribe_audio(audioclip)
        except Exception as e:
            logger.error(f"Transcriber: Error during transcription - {e}")
            return None

        if not transcription:
            return None

        text, language = transcription
        if not text or not text.strip():
            # Silence / VAD-filtered clips come back empty — treat as no speech,
            # or a phantom "I heard:" event wakes the brain.
            return None

        return (text.strip(), language)

    def close(self) -> None:
        """Release the underlying model resources."""
        if self.model is not None and hasattr(self.model, 'close'):
            self.model.close()
            logger.debug("Transcriber: Model released.")
