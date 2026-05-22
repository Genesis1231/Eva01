"""
SpeakerIdentifier — EVA's voice recognition.

Matches speech audio to known voice embeddings using the WeSpeaker
ResNet34-LM model, served via onnxruntime. Features are computed with
kaldi-native-fbank (Kaldi-parity FBANK, no torch). Mirror of
eva/senses/vision/identifier.py for the audio domain.
"""

import hashlib
import pickle
from pathlib import Path
from typing import Dict, Optional

import kaldi_native_fbank as knf
import numpy as np
import onnxruntime as ort
import soundfile as sf
from scipy.spatial.distance import cosine

from config import logger, DATA_DIR
from eva.core.people import PeopleDB
from eva.senses.audio.vad_onnx import SileroVAD, pick_providers


class SpeakerIdentifier:
    """Matches speech to known people by voice embedding similarity."""

    _MODEL_FILE = "wespeaker_resnet34_LM.onnx"
    _CERTAIN_THRESHOLD = 0.35   # cosine distance — below = confident match
    _LIKELY_THRESHOLD = 0.60    # below = likely match
    _MIN_AUDIO_SECONDS = 0.5
    _SAMPLE_RATE = 16000

    # FBANK params — must match WeSpeaker training config.
    # ref: github.com/wenet-e2e/wespeaker conf/resnet.yaml
    _FBANK_MEL_BINS = 80
    _FBANK_FRAME_LENGTH_MS = 25
    _FBANK_FRAME_SHIFT_MS = 10
    # Audio is scaled to int16 range before FBANK — Kaldi convention.
    _WAVEFORM_SCALE = 32768.0

    def __init__(self, people_db: PeopleDB):
        self._people_lookup: Dict[str, str] = people_db.get_id_name_map()
        self.voice_dir = DATA_DIR / "voices"
        self.model_dir = DATA_DIR / "models"
        self._cache_path = self.voice_dir / ".embeddings_cache.pkl"
        self._session: Optional[ort.InferenceSession] = None
        self._input_name: Optional[str] = None
        self._fbank_opts_cached: Optional[knf.FbankOptions] = None
        self._vad: Optional[SileroVAD] = None
        self._initialized = False

        self._ref_embeddings: Dict[str, np.ndarray] = {}

    def init_model(self) -> None:
        """Initialize speaker-identification dependencies once."""
        if self._initialized:
            return

        self._init_model()
        self._init_vad()
        self._load_embeddings()
        self._initialized = True

    def _init_model(self) -> None:
        self.voice_dir.mkdir(parents=True, exist_ok=True)
        model_path = self.model_dir / self._MODEL_FILE
        if not model_path.exists():
            logger.warning(
                f"SpeakerIdentifier: model not found at {model_path}. "
                "Run `eva-setup` to download required models."
            )
            return
        try:
            self._session = ort.InferenceSession(
                str(model_path), providers=pick_providers()
            )
            self._input_name = self._session.get_inputs()[0].name
            self._fbank_opts_cached = self._fbank_opts()
            active = self._session.get_providers()[0]
            logger.debug(f"SpeakerIdentifier: Loaded {self._MODEL_FILE} ({active}).")
        except Exception as e:
            logger.warning(f"SpeakerIdentifier: Failed to load model — {e}")

    def _init_vad(self) -> None:
        try:
            self._vad = SileroVAD()
        except Exception as e:
            logger.warning(f"SpeakerIdentifier: VAD unavailable — {e}")

    def _fbank_opts(self) -> knf.FbankOptions:
        opts = knf.FbankOptions()
        opts.frame_opts.samp_freq = self._SAMPLE_RATE
        opts.frame_opts.frame_length_ms = self._FBANK_FRAME_LENGTH_MS
        opts.frame_opts.frame_shift_ms = self._FBANK_FRAME_SHIFT_MS
        opts.frame_opts.dither = 0.0
        opts.frame_opts.snip_edges = False
        opts.mel_opts.num_bins = self._FBANK_MEL_BINS
        opts.energy_floor = 0.0
        return opts

    def _extract_embedding(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """Run audio (float32 mono 16kHz) through FBANK → WeSpeaker ONNX."""
        if self._session is None:
            return None

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # FBANK on Kaldi int16-range waveform.
        fb = knf.OnlineFbank(self._fbank_opts_cached)
        fb.accept_waveform(self._SAMPLE_RATE, audio * self._WAVEFORM_SCALE)
        fb.input_finished()
        n = fb.num_frames_ready
        if n == 0:
            return None
        feats = np.stack([fb.get_frame(i) for i in range(n)])   # [T, 80]
        feats -= feats.mean(axis=0, keepdims=True)              # per-utterance mean sub
        feats = feats[np.newaxis, ...].astype(np.float32)       # [1, T, 80]

        emb = self._session.run(None, {self._input_name: feats})[0]  # [1, 256]
        return emb.squeeze(0)

    def _trim_silence(self, audio: np.ndarray) -> np.ndarray:
        """Strip leading/trailing silence using Silero VAD."""
        if self._vad is None:
            return audio
        return self._vad.trim_silence(audio)

    def _load_embeddings(self) -> None:
        if self._session is None:
            return

        # Scan voice samples: data/voices/{person_id}/*.wav
        voice_files: Dict[str, list[Path]] = {}
        for person_dir in self.voice_dir.iterdir():
            if not person_dir.is_dir() or person_dir.name.startswith("."):
                continue
            wavs = sorted(person_dir.glob("*.wav"))
            if wavs:
                voice_files[person_dir.name] = wavs

        if not voice_files:
            logger.debug("SpeakerIdentifier: No voice samples found.")
            return

        # Check cache validity via file fingerprint (paths + mtimes).
        # Any schema mismatch (old format, partial write, foreign content) falls
        # through to recompute — cheap, and shields against cross-version drift.
        fingerprint = self._compute_fingerprint(voice_files)
        try:
            with open(self._cache_path, "rb") as f:
                cached = pickle.load(f)
            if (
                isinstance(cached, dict)
                and cached.get("fingerprint") == fingerprint
                and isinstance(cached.get("embeddings"), dict)
            ):
                self._ref_embeddings = cached["embeddings"]
                logger.debug(
                    f"SpeakerIdentifier: Loaded {len(self._ref_embeddings)} voices from cache."
                )
                return
        except (FileNotFoundError, EOFError, pickle.UnpicklingError):
            pass

        # Compute fresh embeddings.
        logger.debug("SpeakerIdentifier: Computing voice embeddings...")
        for person_id, wavs in voice_files.items():
            embeddings = []
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
                        embeddings.append(emb)
                except Exception as e:
                    logger.warning(f"SpeakerIdentifier: Failed on {wav.name} — {e}")
            if embeddings:
                self._ref_embeddings[person_id] = np.mean(embeddings, axis=0)

        # Save cache.
        try:
            with open(self._cache_path, "wb") as f:
                pickle.dump(
                    {"fingerprint": fingerprint, "embeddings": self._ref_embeddings}, f
                )
        except Exception as e:
            logger.warning(f"SpeakerIdentifier: Failed to save cache — {e}")

        logger.debug(
            f"SpeakerIdentifier: Loaded {len(self._ref_embeddings)} known voices."
        )

    @staticmethod
    def _compute_fingerprint(voice_files: Dict[str, list[Path]]) -> str:
        h = hashlib.md5()
        for person_id in sorted(voice_files):
            for wav in voice_files[person_id]:
                h.update(f"{wav}:{wav.stat().st_mtime}".encode())
        return h.hexdigest()

    def identify(self, audio: np.ndarray) -> Optional[Dict]:
        """Identify who is speaking from raw 16kHz mono float32 audio.

        Returns:
            {"id": person_id, "name": name} on match, None otherwise.
        """
        if not self._initialized:
            self.init_model()

        if self._session is None or not self._ref_embeddings:
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

        # Find closest reference.
        best_id = None
        best_dist = float("inf")
        for person_id, ref_emb in self._ref_embeddings.items():
            dist = cosine(embedding, ref_emb)
            if dist < best_dist:
                best_dist = dist
                best_id = person_id

        name = self._people_lookup.get(best_id) if best_id else None
        if not name:
            return None

        if best_dist <= self._CERTAIN_THRESHOLD:
            return {"id": best_id, "name": name}

        if best_dist <= self._LIKELY_THRESHOLD:
            return {"id": best_id, "name": f"Someone sounds like {name}"}

        return None

    def close(self) -> None:
        self._people_lookup.clear()
        self._ref_embeddings.clear()
        self._session = None
        self._input_name = None
        self._fbank_opts_cached = None
        self._vad = None
        self._initialized = False
        logger.debug("SpeakerIdentifier: Released.")
