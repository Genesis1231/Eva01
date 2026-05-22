"""
Record voice samples for speaker identification.

Usage:
    python record_void.py p001
    python record_void.py p001 --list-devices

Records 5 samples with sherpa-onnx Silero VAD to trim silence.
Saves to data/voices/{person_id}/sample_01.wav … sample_05.wav
"""

import argparse
import sys
import time

import numpy as np
import sherpa_onnx
import sounddevice as sd
import soundfile as sf

from config import DATA_DIR

SAMPLE_RATE = 16000
CHANNELS = 1
MAX_RECORD_SECONDS = 12
SILENCE_TIMEOUT = 1.5   # auto-stop after this much trailing silence
MIN_SPEECH_SECONDS = 1.0

PROMPTS = [
    "The quick brown fox jumps over the lazy dog near the riverbank every single morning.",
    "She sells seashells by the seashore while the waves crash against the ancient rocks below.",
    "I believe that technology should serve humanity, not replace it or control the way we live.",
    "Yesterday afternoon, I walked through the park and watched the golden sunset behind the mountains.",
    "Please tell me where the nearest coffee shop is, because I could really use some caffeine right now.",
]


def _build_vad() -> sherpa_onnx.VoiceActivityDetector:
    """Construct a sherpa-onnx Silero VAD with library defaults."""
    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = str(DATA_DIR / "models" / "silero_vad.onnx")
    cfg.sample_rate = SAMPLE_RATE
    return sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=30)


def _trim_with_vad(audio: np.ndarray, vad: sherpa_onnx.VoiceActivityDetector) -> np.ndarray:
    """Concatenate detected speech segments. Returns original if VAD found nothing."""
    vad.reset()
    window = vad.config.silero_vad.window_size
    buf = audio.astype(np.float32, copy=False)
    i = 0
    while i + window <= len(buf):
        vad.accept_waveform(buf[i : i + window])
        i += window
    vad.flush()
    speech: list[float] = []
    while not vad.empty():
        speech.extend(list(vad.front.samples))  # copy before pop()
        vad.pop()
    return np.asarray(speech, dtype=np.float32) if speech else audio


def record_one(index: int, prompt: str) -> np.ndarray | None:
    """Record one sample with VAD-based auto-stop."""
    print(f"\n--- Sample {index + 1}/5 ---")
    print(f"Read this aloud:\n")
    print(f'  "{prompt}"\n')
    input("Press ENTER when ready, then speak... ")

    vad = _build_vad()
    window = vad.config.silero_vad.window_size

    frames: list[np.ndarray] = []
    speech_detected = False
    silent_since: float | None = None
    vad_buffer = np.zeros(0, dtype=np.float32)

    def callback(indata, frame_count, time_info, status):
        nonlocal speech_detected, silent_since, vad_buffer
        if status:
            print(f"  (stream: {status})", file=sys.stderr)
        chunk = indata[:, 0].copy().astype(np.float32)
        frames.append(chunk)

        vad_buffer = np.concatenate([vad_buffer, chunk])
        while len(vad_buffer) >= window:
            vad.accept_waveform(vad_buffer[:window])
            vad_buffer = vad_buffer[window:]
            if vad.is_speech_detected():
                speech_detected = True
                silent_since = None
            elif speech_detected and silent_since is None:
                silent_since = time.time()

    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32",
        blocksize=1024, callback=callback,
    ):
        print("  Recording...", end="", flush=True)
        start = time.time()
        while True:
            time.sleep(0.05)
            elapsed = time.time() - start

            if speech_detected and silent_since and (time.time() - silent_since) >= SILENCE_TIMEOUT:
                print(f" auto-stopped ({elapsed:.1f}s)")
                break
            if elapsed >= MAX_RECORD_SECONDS:
                print(f" max time reached ({MAX_RECORD_SECONDS}s)")
                break

    if not frames:
        print("  No audio captured!")
        return None

    audio = np.concatenate(frames)
    audio = _trim_with_vad(audio, vad)

    duration = len(audio) / SAMPLE_RATE
    if duration < MIN_SPEECH_SECONDS:
        print(f"  Too short ({duration:.2f}s) — skipping")
        return None

    print(f"  Got {duration:.1f}s of speech")
    return audio


def main():
    parser = argparse.ArgumentParser(description="Record voice samples for speaker ID")
    parser.add_argument("person_id", help="Person ID (e.g. p001)")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    out_dir = DATA_DIR / "voices" / args.person_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRecording 5 voice samples for '{args.person_id}'")
    print(f"Saving to: {out_dir}")
    print(f"Auto-stops after {SILENCE_TIMEOUT}s of silence.")

    saved = 0
    for i, prompt in enumerate(PROMPTS):
        audio = record_one(i, prompt)
        if audio is None:
            retry = input("  Retry? [Y/n] ").strip().lower()
            if retry != "n":
                audio = record_one(i, prompt)

        if audio is not None:
            path = out_dir / f"sample_{i + 1:02d}.wav"
            sf.write(str(path), audio, SAMPLE_RATE)
            print(f"  Saved: {path.name}")
            saved += 1

    print(f"\nDone — {saved}/5 samples saved to {out_dir}")
    if saved > 0:
        cache = DATA_DIR / "voices" / ".embeddings_cache.pkl"
        if cache.exists():
            cache.unlink()
            print("Old embeddings cache cleared.")
        print("Restart EVA to load the new voice embeddings.")


if __name__ == "__main__":
    main()
