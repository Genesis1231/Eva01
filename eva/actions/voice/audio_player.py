"""
AudioPlayer: Audio playback for streams and files. 
All methods are blocking — wrap with asyncio.to_thread() in async contexts.

"""
from config import logger
import sounddevice as sd
import mpv

class AudioPlayer:
    """Plays PCM via sounddevice or files/streams via mpv. Methods are blocking."""

    def __init__(self):
        self.player: mpv.MPV | None = None
        self._stop_event: bool = False
        self._current_stream: sd.OutputStream | None = None

    def stop_playback(self) -> None:
        """Stop all active audio playback immediately.

        The sounddevice stream is left alone — play_pcm's loop polls
        _stop_event and exits gracefully; calling abort() here triggers
        ALSA errors.
        """
        self._stop_event = True
        try:
            if self.player:
                self.player.terminate()
                self.player = None
        except Exception:
            logger.warning("AudioPlayer: error while stopping mpv player, it may already be stopped.")

    def play_pcm(self, samples, sample_rate: int) -> None:
        """Play raw PCM (numpy float32 array) via sounddevice. Blocks until done."""
        import numpy as np

        # Honor a stop signalled between synthesis and playback — don't
        # quietly start audio that was already cancelled upstream.
        if self._stop_event:
            self._stop_event = False
            return

        # Ensure samples are float32 numpy array
        if isinstance(samples, list):
            samples = np.array(samples, dtype=np.float32)
        elif isinstance(samples, np.ndarray):
            if samples.dtype != np.float32:
                samples = samples.astype(np.float32)

        # Ensure shape is (frames, channels)
        if samples.ndim == 1:
            samples = samples.reshape(-1, 1)

        chunk_size = 2048  # ~85ms at 24kHz

        try:
            # Use OutputStream for better control
            with sd.OutputStream(samplerate=sample_rate, channels=1, dtype='float32') as stream:
                self._current_stream = stream

                total_frames = len(samples)
                for i in range(0, total_frames, chunk_size):
                    if self._stop_event:
                        break

                    # Calculate chunk
                    end_frame = min(i + chunk_size, total_frames)
                    chunk = samples[i:end_frame]

                    stream.write(chunk)

        except Exception as e:
            logger.warning(f"AudioPlayer: error while playing PCM audio: {e}")
        finally:
            self._stop_event = False
            self._current_stream = None
         
    def play_stream(self, path: str) -> None:
        """Blocking: plays file/url via mpv."""
        if not path:
            return
        if self._stop_event:
            self._stop_event = False
            return
        try:
            self.player = mpv.MPV()
            self.player.play(path)
            self.player.wait_for_playback()
        except Exception as e:
            logger.error(f"AudioPlayer: error while playing stream: {e}")
            raise Exception(f"Error: Failed to play stream: {e}")
        finally:
            self._stop_event = False
            if self.player:
                self.player.terminate()
                self.player = None
    

            

