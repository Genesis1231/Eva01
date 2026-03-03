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

    def stop_playback(self) -> None:
        """Stop all active audio playback immediately."""
        self._stop_event = True

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
        sd.play(samples, sample_rate)
        sd.wait()
         
    def play_stream(self, path: str) -> None:
        """Blocking: plays file/url via mpv. Wrap with asyncio.to_thread() in async contexts."""
        if not path:
            return
        try:
            self.player = mpv.MPV()
            self.player.play(path)
            self.player.wait_for_playback()
            self.player.terminate()
            self.player = None
        except Exception as e:
            raise Exception(f"Error: Failed to play stream: {e}")
    

            

