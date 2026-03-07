from pathlib import Path
from typing import List, Dict

import numpy as np
from deepface import DeepFace

from config import logger, DATA_DIR
from eva.core.people import PeopleDB


class Identifier:
    """EVA's facial recognition — matches faces to people she knows."""

    _MODEL_NAME = "Facenet512"
    _DETECTOR_BACKEND = "retinaface"
    _CERTAIN_DISTANCE_THRESHOLD = 0.20  # Below this, we are quite confident in the match.
    _LIKELY_DISTANCE_THRESHOLD = 0.40 # Below this, we think it's likely but not certain.

    def __init__(self, people_db: PeopleDB):
        # Snapshot the only data Identifier needs, then avoid PeopleDB at runtime.
        self._people_lookup: Dict[str, str] = people_db.get_id_name_map()
        self._db_path = DATA_DIR / "faces"
        self._init_model()

    def _init_model(self) -> None:
        """Initialize the recognition model."""
        self._db_path.mkdir(parents=True, exist_ok=True)
        try:
            # Pre-load the model to avoid delay on first use
            DeepFace.build_model(model_name=self._MODEL_NAME)
            logger.debug(
                f"Identifier: Loaded {self._MODEL_NAME} with {len(self._people_lookup)} known people."
            )
        except Exception as e:
            logger.warning(f"Identifier: Failed to load identification model — {e}")

    def identify(self, frame: np.ndarray) -> List[Dict]:
        """Identify faces in a frame.

        Returns:
            List of dicts with keys: id, name.
        """
        
        # Quick check if we have any faces to match against
        # This is faster than iterating through the database
        logger.debug("Identifier: Starting identification process.")
        if not any(self._db_path.iterdir()):
            logger.warning("Identifier: No faces in database to match.")
            return []

        try:
            dfs = DeepFace.find(
                img_path=frame,
                db_path=str(self._db_path),
                model_name=self._MODEL_NAME,
                detector_backend=self._DETECTOR_BACKEND,
                enforce_detection=False,
                silent=True,
            )
        except Exception as e:
            logger.error(f"Identifier: Recognition error — {e}")
            return []

        results = []
        for df in dfs:
            if df.empty:  # type: ignore
                # Face detected but no match — a stranger
                results.append(
                    {
                        "id": None,
                        "name": "someone I don't recognize",
                    }
                )
                continue

            # Get the best match (first row)
            match = df.iloc[0]  # type: ignore
            identity_path = Path(match["identity"])

            # The person_id is the parent directory name
            # Structure: data/faces/{person_id}/image.jpg
            person_id = identity_path.parent.name

            name = self._people_lookup.get(person_id)
            if not name:
                results.append(
                    {
                        "id": None,
                        "name": "someone I don't recognize",
                    }
                )
                continue

            distance = float(match.get("distance", 1.0))
            logger.debug(f"Identifier: Identification results — {name} (id: {person_id}), distance: {distance:.4f}")
            if distance <= self._CERTAIN_DISTANCE_THRESHOLD:
                results.append(
                    {
                        "id": person_id,
                        "name": name,
                    }
                )
                continue

            if distance <= self._LIKELY_DISTANCE_THRESHOLD:
                # Keep id None for likely matches to avoid contaminating memory updates.
                results.append(
                    {
                        "id": None,
                        "name": f"looks like {name}",
                    }
                )
                continue

            results.append(
                {
                    "id": None,
                    "name": "someone I don't recognize",
                }
            )

        return results

    def close(self) -> None:
        """Release the face recognition model and TF session from memory."""
        from deepface.modules import modeling

        self._people_lookup.clear()
        if hasattr(modeling, "cached_models"):
            modeling.cached_models.clear()

        try:
            import keras
            keras.backend.clear_session()
        except Exception:
            pass

        logger.debug(f"Identifier: Model {self._MODEL_NAME} released.")
