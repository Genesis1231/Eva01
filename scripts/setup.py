"""Provision EVA's ONNX models. Run via `eva-setup` after install."""

import argparse
import hashlib
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from config import DATA_DIR, logger

MODELS_DIR = DATA_DIR / "models"


@dataclass(frozen=True)
class ModelSpec:
    """A single ONNX model to provision: where it lives, where to fetch, expected SHA."""
    label: str
    path: Path
    url: str
    sha256: str


SILERO_VAD = ModelSpec(
    label="Silero VAD",
    path=MODELS_DIR / "silero_vad.onnx",
    url=(
        "https://raw.githubusercontent.com/snakers4/silero-vad/v6.2.1/"
        "src/silero_vad/data/silero_vad.onnx"
    ),
    sha256="1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3",
)

# sherpa-onnx's tagged release — same WeSpeaker weights with `model_type`
# metadata baked in, which sherpa_onnx.SpeakerEmbeddingExtractor requires.
WESPEAKER_RESNET34_LM = ModelSpec(
    label="WeSpeaker ResNet34-LM",
    path=MODELS_DIR / "wespeaker_resnet34_LM.onnx",
    url=(
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
        "speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx"
    ),
    sha256="e9848563da86f263117134dfd7ad63c92355b37de492b55e325400c9d9c39012",
)

MODELS = [SILERO_VAD, WESPEAKER_RESNET34_LM]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _download_verified(url: str, dest: Path, expected_sha256: str) -> None:
    """Download `url` to `dest` via a .tmp file, verify SHA, then atomic rename.

    Atomic rename ensures `dest` only ever holds a complete + verified file —
    a killed/partial download leaves only the .tmp behind, never a corrupt final.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    tmp.unlink(missing_ok=True)

    logger.info(f"Downloading {url}")
    try:
        urllib.request.urlretrieve(url, tmp)
    except urllib.error.URLError as e:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download {url}") from e

    actual = _sha256(tmp)
    if actual != expected_sha256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"SHA-256 mismatch for {dest.name}: expected {expected_sha256}, got {actual}"
        )

    tmp.replace(dest)


def ensure_model(spec: ModelSpec, force: bool = False) -> Path:
    """Provision one model: re-download if missing, corrupt, or `force=True`."""
    if force:
        spec.path.unlink(missing_ok=True)

    if spec.path.exists():
        if _sha256(spec.path) == spec.sha256:
            logger.info(f"{spec.label}: OK at {spec.path}")
            return spec.path
        logger.warning(f"{spec.label}: SHA mismatch at {spec.path}, re-downloading")
        spec.path.unlink()

    _download_verified(spec.url, spec.path, spec.sha256)
    logger.info(f"{spec.label}: installed at {spec.path}")
    return spec.path


def download_all_models(force: bool = False) -> None:
    logger.info("EVA setup: provisioning models")
    total = len(MODELS)
    for i, spec in enumerate(MODELS, start=1):
        logger.info(f"[{i}/{total}] {spec.label}")
        ensure_model(spec, force=force)
    logger.info("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up EVA models")
    parser.add_argument("--force", action="store_true", help="Re-download even if present")
    args = parser.parse_args()
    download_all_models(force=args.force)


if __name__ == "__main__":
    main()
