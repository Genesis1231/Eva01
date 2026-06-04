"""Eva's embedding engine — one async facade over her multimodal embedding model(s).

Two swappable providers, selected by a `provider:model` spec:
    "qwenvl:Qwen3-VL-Embedding-8B"   — local MLX server: text + image embeddings AND the raw
                                       patch-token grid (the vision L1 signal). Default.
    "gemini:gemini-embedding-2"      — cloud Gemini Embedding 2: text + image embeddings (no patch).

The engine is Eva's single Qwen-VL client: `embed_one`/`embed_many` (text), `embed_image`, and
`patch` (raw token grid) all hit the same model. Patch features are Qwen-only — Gemini has no patch
endpoint, so `patch()` returns None there (guarded by `supports_patches`). All calls safe-degrade to
None when the provider is unavailable, so a dead server never raises into the caller.
"""

from __future__ import annotations

import asyncio
import base64
from typing import Any

import numpy as np

from config import logger


def _data_uri(image: bytes, mime: str) -> str:
    """Encoded image bytes -> a base64 data-URI the embedding servers accept."""
    return f"data:{mime};base64," + base64.b64encode(image).decode("ascii")


class QwenVLEmbedding:
    """Local Qwen3-VL-Embedding MLX server: text + image embeddings and the raw patch-token grid.
    """

    supports_patches = True

    def __init__(self, model: str, base_url: str | None, dimensions: int = 1024) -> None:
        self.model = model
        self.base_url = (base_url or "http://192.168.3.46:8000").rstrip("/")
        self.dimensions = dimensions
        self._client: Any = None

    def init(self) -> None:
        import httpx  # transport dep (already present via eva/utils/feed.py)

        with httpx.Client(trust_env=False, timeout=5.0) as client:
            client.get(f"{self.base_url}/health").raise_for_status()

    def _aclient(self):
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(base_url=self.base_url, trust_env=False, timeout=30.0)
        return self._client

    async def _post(self, path: str, payload: dict) -> dict:
        response = await self._aclient().post(path, json=payload)
        response.raise_for_status()
        return response.json()

    async def _embeddings(self, inputs: list[dict]) -> list[list[float]]:
        response = await self._post(
            "/v1/embeddings",
            {"inputs": inputs, "dimensions": self.dimensions, "normalize": True},
        )
        return [list(item["embedding"]) for item in response["data"]]

    async def embed(self, text: str) -> list[float]:
        return (await self._embeddings([{"text": text}]))[0]

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        return await self._embeddings([{"text": t} for t in texts])

    async def embed_image(self, image: bytes, mime: str) -> list[float]:
        return (await self._embeddings([{"image": _data_uri(image, mime)}]))[0]

    async def patch(self, image: bytes, mime: str) -> Optional[np.ndarray]:
        """A frame -> its unit-norm patch tokens (PP, FD), full native dim. None if non-finite."""
        response = await self._post(
            "/v1/patch_features",
            {"inputs": [{"image": _data_uri(image, mime)}], "patch_grid": 16, "normalize": True},
        )
        grid = response["data"][0]["grid"]
        array = np.frombuffer(base64.b64decode(grid["data"]), np.float16).astype(np.float32)
        array = array.reshape(-1, grid["shape"][-1])              # patch grid -> (patches, FD)
        if not np.isfinite(array).all():
            return None
        return array / (np.linalg.norm(array, axis=1, keepdims=True) + 1e-9)


class GeminiEmbedding:
    """Cloud Gemini Embedding 2: text + image embeddings (Matryoshka, default 768-d). No patch grid.

    The google-genai client is synchronous, so calls run in a worker thread. Auth via the
    GOOGLE_API_KEY / GEMINI_API_KEY environment variable (picked up by genai.Client())."""

    supports_patches = False

    def __init__(self, model: str, dimensions: int = 768) -> None:
        self.model = model
        self.dimensions = dimensions
        self._client: Any = None

    def init(self) -> None:
        try:
            from google import genai  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError("Install optional dependency 'google-genai' for the gemini embedder") from e

        self._client = genai.Client()

    async def _embed_content(self, contents: Any) -> list[list[float]]:
        from google.genai import types  # type: ignore[import-not-found]

        config = types.EmbedContentConfig(output_dimensionality=self.dimensions)

        def _run() -> list[list[float]]:
            response = self._client.models.embed_content(
                model=self.model, contents=contents, config=config
            )
            return [list(e.values) for e in response.embeddings]

        return await asyncio.to_thread(_run)

    async def embed(self, text: str) -> list[float]:
        return (await self._embed_content(text))[0]

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        return await self._embed_content(list(texts))

    async def embed_image(self, image: bytes, mime: str) -> list[float]:
        from google.genai import types  # type: ignore[import-not-found]

        part = types.Part.from_bytes(data=image, mime_type=mime)
        return (await self._embed_content(part))[0]

    async def patch(self, image: bytes, mime: str) -> Optional[np.ndarray]:
        return None  # Gemini has no patch-feature endpoint


class EmbeddingEngine:
    """Provider-agnostic multimodal embedding facade with safe degradation."""

    def __init__(self, embedding_provider: str, base_url: Optional[str] = None) -> None:
        self.embedding_provider = embedding_provider or "qwenvl:Qwen3-VL-Embedding-8B"
        self.base_url = base_url

        provider, model = self._parse_provider_spec(self.embedding_provider)
        self.provider = provider
        self.model = model

        self._enabled = False
        self._dimension: Optional[int] = None
        self._embedding: Optional[QwenVLEmbedding | GeminiEmbedding] = None
        self._supports_patches = False

    @staticmethod
    def _parse_provider_spec(provider_spec: str) -> tuple[str, str]:
        if ":" not in provider_spec:
            raise ValueError(
                "Embedding provider must use format 'provider:model', "
                f"got '{provider_spec}'"
            )
        provider, model = provider_spec.split(":", 1)
        return provider.strip().lower(), model.strip()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def dimension(self) -> Optional[int]:
        return self._dimension

    @property
    def multimodal(self) -> bool:
        """Both supported providers embed images; True once successfully initialized."""
        return self._enabled and self._embedding is not None

    @property
    def supports_patches(self) -> bool:
        return self._supports_patches

    def provider_id(self) -> str:
        return self.provider

    def model_id(self) -> str:
        return self.model

    def init_model(self) -> None:
        """Initialize the provider. Never raises; disables on failure (dead server, missing dep)."""
        try:
            if self.provider == "qwenvl":
                self._embedding = QwenVLEmbedding(self.model, base_url=self.base_url)
            elif self.provider == "gemini":
                self._embedding = GeminiEmbedding(self.model)
            else:
                raise ValueError(f"Unsupported embedding provider: {self.provider}")

            self._embedding.init()
            self._supports_patches = self._embedding.supports_patches
            self._enabled = True
            logger.debug(f"EmbeddingEngine: initialized {self.provider}:{self.model}")
        except Exception as e:
            self._enabled = False
            logger.warning(f"EmbeddingEngine {self.provider} error - {e}")

    async def embed_one(self, text: str) -> Optional[list[float]]:
        """Return embedding vector for text, or None when disabled/failing."""
        if not self._enabled or not self._embedding:
            return None

        if not text or not text.strip():
            logger.warning("EmbeddingEngine: empty text, skipping embedding")
            return None

        try:
            vector = await self._embedding.embed(text)
            if self._dimension is None:
                self._dimension = len(vector)
            return vector
        except Exception as e:
            logger.warning(f"EmbeddingEngine: {self.provider} embedding failed - {e}")
            return None

    async def embed_many(self, texts: list[str]) -> list[Optional[list[float]]]:
        """Embed multiple texts. Returns list parallel to input; None for empty/failed."""
        if not self._enabled or not self._embedding:
            return [None] * len(texts)

        valid_indices = [i for i, t in enumerate(texts) if t and t.strip()]
        if not valid_indices:
            return [None] * len(texts)

        try:
            vectors = await self._embedding.embed_many([texts[i] for i in valid_indices])
            result: list[Optional[list[float]]] = [None] * len(texts)
            for idx, vec in zip(valid_indices, vectors):
                result[idx] = vec
                if self._dimension is None and vec:
                    self._dimension = len(vec)
            return result
        except Exception as e:
            logger.warning(f"EmbeddingEngine: batch embed failed — {e}")
            return [None] * len(texts)

    async def embed_image(self, image: bytes, mime: str = "image/jpeg") -> Optional[list[float]]:
        """Return the pooled embedding for an encoded image (bytes), or None when disabled/failing."""
        if not self._enabled or not self._embedding or not image:
            return None
        try:
            vector = await self._embedding.embed_image(image, mime)
            if vector and self._dimension is None:
                self._dimension = len(vector)
            return vector
        except Exception as e:
            logger.warning(f"EmbeddingEngine: {self.provider} image embedding failed - {e}")
            return None

    async def patch(self, image: bytes, mime: str = "image/jpeg") -> Optional[np.ndarray]:
        """Return the raw (PP, FD) patch-token grid for an encoded image, or None when unsupported/failing."""
        if not self._enabled or not self._embedding or not self._supports_patches or not image:
            return None
        try:
            return await self._embedding.patch(image, mime)
        except Exception as e:
            logger.warning(f"EmbeddingEngine: {self.provider} patch failed - {e}")
            return None
