"""Visual novelty scoring — the pure-numpy scorers + small frame/vector helpers.

Learned patch tokens and pooled embeddings come from the shared EmbeddingEngine
(eva/database/embeddings.py). This module turns frames into bytes, provides a CV2 fallback patch grid,
shapes pooled embeddings, and scores a representation vs a reference set — one scorer per timescale,
both using the same k=KNN smoothing:

    L1 (patch): `patch_novelty` scores a frame's patch tokens vs a reference patch set — mean of the
                TOP_K most-novel patches, each patch via per-patch k-NN.
    L2 (embed): `embed_novelty` scores a frame's pooled vector vs a reference set — 1 - mean of the
                KNN nearest cosines.
"""

import cv2
import numpy as np

KNN, TOP_K = 8, 5    # per-patch k-NN; frame novelty = mean of the TOP_K most-novel patches
CV_GRID_SIZE = (320, 240)     # width, height
CV_GRID_TILES = (8, 8)      # rows, columns


def to_jpeg(frame: np.ndarray, quality: int = 88) -> bytes:
    """A webcam frame -> encoded JPEG bytes for the embedding engine."""
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return encoded.tobytes() if ok else b""


def as_vector(embedding: list[float] | None) -> np.ndarray | None:
    """The engine's pooled embedding (a list) -> a unit-norm (1, D) row, or None."""
    if not embedding:
        return None
    vector = np.asarray(embedding, np.float32)
    return (vector / (np.linalg.norm(vector) + 1e-9))[None, :]


def cv_patch_grid(frame: np.ndarray) -> np.ndarray:
    """CV2 fallback patch grid for providers without learned patch tokens."""
    
    small = cv2.resize(frame, CV_GRID_SIZE, interpolation=cv2.INTER_AREA)
    if small.ndim == 3:
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    else:
        gray = small
    gray = cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32) / 255.0

    rows, columns = CV_GRID_TILES
    tile_h = CV_GRID_SIZE[1] // rows
    tile_w = CV_GRID_SIZE[0] // columns
    patches = []
    for row in range(rows):
        y0 = row * tile_h
        for column in range(columns):
            x0 = column * tile_w
            tile = gray[y0:y0 + tile_h, x0:x0 + tile_w].reshape(-1) - 0.5
            norm = float(np.linalg.norm(tile))
            if norm < 1e-6:
                tile = np.full(tile.shape, 1.0 / np.sqrt(tile.size), dtype=np.float32)
            else:
                tile = tile / norm
            patches.append(tile.astype(np.float32, copy=False))
    
    return np.vstack(patches).astype(np.float32, copy=False)


# L1: per-patch novelty (recency-FIFO habituation)
def patch_novelty(query_patches: np.ndarray, reference_patches: np.ndarray) -> float:
    """Frame novelty of query_patches vs a reference patch set: mean of the TOP_K most-novel query
    patches, each patch's novelty = mean cosine distance to its KNN nearest neighbours."""

    similarities = query_patches @ reference_patches.T
    k = min(KNN, similarities.shape[1])
    patch_distances = 1.0 - np.partition(similarities, -k, axis=1)[:, -k:].mean(axis=1)
    return float(np.sort(patch_distances)[-TOP_K:].mean())


# L2: recognition novelty (lifelong recognition)
def embed_novelty(query: np.ndarray, reference: np.ndarray) -> float:
    """Novelty of a frame's pooled embedding (1, D) vs a reference set of normal vectors:
    1 - mean of the KNN nearest cosines. (1-NN over a small bank is spiky — one lucky match
    zeroes the score — so k-NN is a steadier estimate and matches L1's smoothing.)"""

    similarities = reference @ query[0]
    k = min(KNN, similarities.shape[0])
    return float(1.0 - np.partition(similarities, -k)[-k:].mean())
