"""
AudioPlayer: Audio playback for streams and files. All methods are blocking — wrap with asyncio.to_thread() in async contexts.
  play_stream(path) -> plays file/url via mpv, blocks until done
  play_pcm(samples, sample_rate) -> plays raw PCM via sounddevice
  stop_playback() -> stops any playing audio
"""

from typing import Optional

import sounddevice as sd
import mpv

class AudioPlayer:
    """
    A class to play audio data.
    
    Attributes:
    device (str): The device used to play audio.
    sample_rate (int): The sample rate of the audio data.
    speaking (bool): A flag to indicate if audio is currently speaking.
    audio_thread (threading.Thread): A thread to play audio.
    
    Methods:
    play_audio: Play audio data from a file or numpy array.
    play_mp3_stream: Play an mp3 stream.
    play_stream: Play an mp3 url.
    stop_playback: Interrupt all playback.
    """
    def __init__(self):
        self.player: Optional[mpv.MPV] = None
        self._stop_event: bool = False
        self._current_stream: Optional[sd.OutputStream] = None

    def stop_playback(self) -> None:
        """Stop all active audio playback immediately."""
        self._stop_event = True
        
        # Stop sounddevice stream if any
        if self._current_stream:
            # We don't abort() anymore to avoid ALSA errors.
            # The play_pcm loop will catch _stop_event and exit gracefully.
            pass

        # Stop mpv python player
        try:
            # Recreate player to fully stop current playback
            if hasattr(self, 'player') and self.player:
                self.player.terminate()
                self.player = None
        except Exception:
            pass

    def play_pcm(self, samples, sample_rate: int) -> None:
        """Play raw PCM (numpy float32 array) via sounddevice. Blocks until done."""
        import numpy as np
        self._stop_event = False
        
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
            # If aborted, this is expected
            pass
        finally:
            self._current_stream = None
         
    def play_stream(self, path: str) -> None:
        """Blocking: plays file/url via mpv."""
        if not path:
            return
        try:
            self.player = mpv.MPV()
            self.player.play(path)
            self.player.wait_for_playback()
        except Exception as e:
            raise Exception(f"Error: Failed to play stream: {e}")
        finally:
            if self.player:
                self.player.terminate()
                self.player = None
    

            

