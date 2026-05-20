"""
Record voice samples for speaker identification.

Usage:
    python record_void.py p001
    python record_void.py p001 --list-devices

Records 5 samples with Silero VAD to trim silence.
Saves to data/voices/{person_id}/sample_01.wav … sample_05.wav
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from silero_vad import load_silero_vad, get_speech_timestamps

SAMPLE_RATE = 16000
CHANNELS = 1
MAX_RECORD_SECONDS = 12
SILENCE_TIMEOUT = 1.5   # auto-stop after this much trailing silence
MIN_SPEECH_SECONDS = 1.0

DATA_DIR = Path(__file__).resolve().parent / "data"

PROMPTS = [
    "The quick brown fox jumps over the lazy dog near the riverbank every single morning.",
    "She sells seashells by the seashore while the waves crash against the ancient rocks below.",
    "I believe that technology should serve humanity, not replace it or control the way we live.",
    "Yesterday afternoon, I walked through the park and watched the golden sunset behind the mountains.",
    "Please tell me where the nearest coffee shop is, because I could really use some caffeine right now.",
]


def trim_silence(audio: np.ndarray, vad_model) -> np.ndarray:
    """Trim leading/trailing silence using Silero VAD."""
    tensor = torch.from_numpy(audio).float()
    stamps = get_speech_timestamps(tensor, vad_model, sampling_rate=SAMPLE_RATE)
    if not stamps:
        return audio

    start = max(0, stamps[0]["start"] - SAMPLE_RATE // 10)   # 100ms padding
    end = min(len(audio), stamps[-1]["end"] + SAMPLE_RATE // 10)
    return audio[start:end]


def record_one(index: int, prompt: str, vad_model) -> np.ndarray | None:
    """Record one sample with VAD-based auto-stop."""
    print(f"\n--- Sample {index + 1}/5 ---")
    print(f"Read this aloud:\n")
    print(f'  "{prompt}"\n')
    input("Press ENTER when ready, then speak... ")

    frames: list[np.ndarray] = []
    speech_detected = False
    silent_since: float | None = None

    # Use a small Silero window to check speech on-the-fly
    vad_chunk_size = 512  # Silero expects 512 samples at 16kHz (32ms)
    vad_buffer = np.zeros(0, dtype=np.float32)

    def callback(indata, frame_count, time_info, status):
        nonlocal speech_detected, silent_since, vad_buffer
        if status:
            print(f"  (stream: {status})", file=sys.stderr)
        chunk = indata[:, 0].copy().astype(np.float32)
        frames.append(chunk)

        # Accumulate for VAD
        vad_buffer = np.concatenate([vad_buffer, chunk])
        while len(vad_buffer) >= vad_chunk_size:
            window = torch.from_numpy(vad_buffer[:vad_chunk_size])
            prob = vad_model(window, SAMPLE_RATE).item()
            vad_buffer = vad_buffer[vad_chunk_size:]

            if prob > 0.5:
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

    # Reset VAD state for trim pass
    vad_model.reset_states()
    audio = trim_silence(audio, vad_model)
    vad_model.reset_states()

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

    print("Loading Silero VAD...")
    vad_model = load_silero_vad()

    print(f"\nRecording 5 voice samples for '{args.person_id}'")
    print(f"Saving to: {out_dir}")
    print(f"Auto-stops after {SILENCE_TIMEOUT}s of silence.")

    saved = 0
    for i, prompt in enumerate(PROMPTS):
        audio = record_one(i, prompt, vad_model)
        if audio is None:
            retry = input("  Retry? [Y/n] ").strip().lower()
            if retry != "n":
                audio = record_one(i, prompt, vad_model)

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
