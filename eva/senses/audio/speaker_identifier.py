"""
SpeakerIdentifier — EVA's voice recognition.

Matches speech audio to known voice embeddings using the WeSpeaker
ResNet34-LM model, served via sherpa-onnx. The library owns FBANK feature
extraction, ONNX inference, and the embedding registry — we just plug in
the model file and the reference voice samples.
"""

import hashlib
import pickle
from pathlib import Path

import numpy as np
import sherpa_onnx
import soundfile as sf

from config import logger, DATA_DIR
from eva.core.people import PeopleDB


class SpeakerIdentifier:
    """Matches speech to known people by voice embedding similarity."""

    _MODEL_FILE = "wespeaker_resnet34_LM.onnx"

    # Cosine similarity thresholds, tuned against measured score distribution:
    # real speaker (hold-one-out) min ≈ 0.83, pitch-shifted impostor ≈ 0.41,
    # white-noise / sine-sweep ≈ 0.30, silence ≈ 0.10.
    _CERTAIN_THRESHOLD = 0.65   # ≥ this = confident match
    _LIKELY_THRESHOLD = 0.45    # ≥ this = "sounds like"

    # Below ~1.5s the embedding is unstable: real speaker and white noise both
    # land near 0.36 against the reference — indistinguishable. We refuse to
    # score short clips rather than guess.
    _MIN_AUDIO_SECONDS = 2.0
    _SAMPLE_RATE = 16000
    _CACHE_FORMAT = "sherpa-onnx-v1"

    def __init__(self, people_db: PeopleDB):
        self._people_lookup: dict[str, str] = people_db.get_id_name_map()
        self._extractor: sherpa_onnx.SpeakerEmbeddingExtractor | None = None
        self._manager: sherpa_onnx.SpeakerEmbeddingManager | None = None
        self._vad: sherpa_onnx.VoiceActivityDetector | None = None
        self._initialized = False

    def init_model(self) -> None:
        """Initialize speaker-identification dependencies once."""
        if self._initialized:
            return
        self._init_extractor()
        if self._extractor is not None:
            self._init_vad()
            self._load_embeddings()
        self._initialized = True

    def _init_extractor(self) -> None:
        model_dir = DATA_DIR / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / self._MODEL_FILE
        if not model_path.exists():
            logger.warning(
                f"SpeakerIdentifier: model not found at {model_path}. "
                "Run `eva-setup` to download required models."
            )
            return
        try:
            cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(model_path), provider="cpu"
            )
            self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
            self._manager = sherpa_onnx.SpeakerEmbeddingManager(self._extractor.dim)
            logger.debug(f"SpeakerIdentifier: Loaded {self._MODEL_FILE}.")
        except Exception as e:
            logger.warning(f"SpeakerIdentifier: Failed to load model — {e}")

    def _init_vad(self) -> None:
        """Load the Silero VAD used to strip leading/trailing silence."""
        model_path = DATA_DIR / "models" / "silero_vad.onnx"
        if not model_path.exists():
            logger.warning(
                f"SpeakerIdentifier: VAD model not found at {model_path}. "
                "Identifying without silence trim — accuracy may degrade on padded clips."
            )
            return
        try:
            cfg = sherpa_onnx.VadModelConfig()
            cfg.silero_vad.model = str(model_path)
            cfg.sample_rate = self._SAMPLE_RATE
            self._vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=30)
        except Exception as e:
            logger.warning(f"SpeakerIdentifier: Failed to load VAD — {e}")

    def _trim_silence(self, audio: np.ndarray) -> np.ndarray:
        """Concatenate detected speech segments. Returns original if VAD unavailable."""
        if self._vad is None:
            return audio
        self._vad.reset()
        window = self._vad.config.silero_vad.window_size
        buf = audio.astype(np.float32, copy=False)
        i = 0
        while i + window <= len(buf):
            self._vad.accept_waveform(buf[i : i + window])
            i += window
        self._vad.flush()
        speech: list[float] = []
        while not self._vad.empty():
            speech.extend(list(self._vad.front.samples))  # copy before pop()
            self._vad.pop()
        return np.asarray(speech, dtype=np.float32) if speech else audio

    def _extract_embedding(self, audio: np.ndarray) -> np.ndarray | None:
        """Run audio (float32 mono 16 kHz) through the sherpa extractor."""
        assert self._extractor is not None, "init_model() must run first"
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        stream = self._extractor.create_stream()
        stream.accept_waveform(sample_rate=self._SAMPLE_RATE, waveform=audio)
        stream.input_finished()
        if not self._extractor.is_ready(stream):
            logger.warning("SpeakerIdentifier: stream not ready for compute.")
            return None
        return np.array(self._extractor.compute(stream), dtype=np.float32)

    def _load_embeddings(self) -> None:
        """Scan voice samples, compute embeddings, register with the manager."""
        voice_dir = DATA_DIR / "voices"
        voice_dir.mkdir(parents=True, exist_ok=True)
        cache_path = voice_dir / ".embeddings_cache.pkl"

        # Scan voice samples: data/voices/{person_id}/*.wav
        voice_files: dict[str, list[Path]] = {}
        for person_dir in voice_dir.iterdir():
            if not person_dir.is_dir() or person_dir.name.startswith("."):
                continue
            wavs = sorted(person_dir.glob("*.wav"))
            if wavs:
                voice_files[person_dir.name] = wavs

        if not voice_files:
            logger.debug("SpeakerIdentifier: No voice samples found.")
            return

        fingerprint = self._compute_fingerprint(voice_files)
        embeddings = self._load_cache(cache_path, fingerprint)
        if embeddings is None:
            embeddings = self._compute_embeddings(voice_files)
            self._save_cache(cache_path, fingerprint, embeddings)

        for person_id, emb in embeddings.items():
            self._manager.add(person_id, emb)

        logger.debug(
            f"SpeakerIdentifier: Loaded {self._manager.num_speakers} known voices."
        )

    def _load_cache(
        self, cache_path: Path, fingerprint: str
    ) -> dict[str, np.ndarray] | None:
        """Return cached embeddings if fingerprint+format match, else None."""
        try:
            with open(cache_path, "rb") as f:
                cached = pickle.load(f)
            if (
                isinstance(cached, dict)
                and cached.get("format") == self._CACHE_FORMAT
                and cached.get("fingerprint") == fingerprint
                and isinstance(cached.get("embeddings"), dict)
            ):
                return cached["embeddings"]
        except (FileNotFoundError, EOFError, pickle.UnpicklingError):
            pass
        return None

    def _save_cache(
        self,
        cache_path: Path,
        fingerprint: str,
        embeddings: dict[str, np.ndarray],
    ) -> None:
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(
                    {
                        "format": self._CACHE_FORMAT,
                        "fingerprint": fingerprint,
                        "embeddings": embeddings,
                    },
                    f,
                )
        except Exception as e:
            logger.warning(f"SpeakerIdentifier: Failed to save cache — {e}")

    def _compute_embeddings(
        self, voice_files: dict[str, list[Path]]
    ) -> dict[str, np.ndarray]:
        """Mean-of-N embedding per person_id across their voice samples."""
        result: dict[str, np.ndarray] = {}
        for person_id, wavs in voice_files.items():
            embs: list[np.ndarray] = []
            for wav in wavs:
                try:
                    audio, sr = sf.read(str(wav), dtype="float32")
                    if sr != self._SAMPLE_RATE:
                        logger.warning(
                            f"SpeakerIdentifier: {wav.name} sample rate {sr} != "
                            f"{self._SAMPLE_RATE}, skipping"
                        )
                        continue
                    if audio.ndim > 1:
                        audio = audio.mean(axis=1)
                    emb = self._extract_embedding(audio)
                    if emb is not None:
                        embs.append(emb)
                except Exception as e:
                    logger.warning(f"SpeakerIdentifier: Failed on {wav.name} — {e}")
            if embs:
                result[person_id] = np.mean(embs, axis=0)
        return result

    @staticmethod
    def _compute_fingerprint(voice_files: dict[str, list[Path]]) -> str:
        h = hashlib.md5()
        for person_id in sorted(voice_files):
            for wav in voice_files[person_id]:
                h.update(f"{wav}:{wav.stat().st_mtime}".encode())
        return h.hexdigest()

    def identify(self, audio: np.ndarray) -> dict[str, str] | None:
        """Identify who is speaking from raw 16 kHz mono float32 audio."""

        if self._manager is None or self._manager.num_speakers == 0:
            return None

        audio = self._trim_silence(audio)
        if len(audio) < self._SAMPLE_RATE * self._MIN_AUDIO_SECONDS:
            return None

        try:
            embedding = self._extract_embedding(audio)
        except Exception as e:
            logger.warning(f"SpeakerIdentifier: Embedding failed — {e}")
            return None
        if embedding is None:
            return None

        # search() runs the nearest-neighbour scan in C++ and returns the
        # best match above threshold, or "" if none qualifies.
        best_id = self._manager.search(embedding, threshold=self._LIKELY_THRESHOLD)
        if not best_id:
            return None

        name = self._people_lookup.get(best_id)
        if not name:
            return None

        if self._manager.score(best_id, embedding) >= self._CERTAIN_THRESHOLD:
            return {"id": best_id, "name": name}
        
        return {"id": best_id, "name": f"Someone sounds like {name}"}

    def close(self) -> None:
        self._people_lookup.clear()
        self._extractor = None
        self._manager = None
        self._vad = None
        self._initialized = False
        logger.debug("SpeakerIdentifier: Released.")
