from pathlib import Path
from typing import List, Dict

import numpy as np
from deepface import DeepFace

from config import logger
from config import DATA_DIR
from eva.core.people import PeopleDB


class Identifier:
    """EVA's facial recognition — matches faces to people she knows."""

    _MODEL_NAME = "Facenet512"
    _DETECTOR_BACKEND = "opencv"

    def __init__(self, people_db: PeopleDB):
        self.people = people_db
        self._db_path = DATA_DIR / "faces"
        self._init_model()

    def _init_model(self):
        """Initialize the recognition model."""
        self._db_path.mkdir(parents=True, exist_ok=True)
        try:
            # Pre-load the model to avoid delay on first use
            DeepFace.build_model(model_name=self._MODEL_NAME)
            logger.debug(f"Identifier: Loaded {self._MODEL_NAME}.")
        except Exception as e:
            logger.warning(f"Identifier: Failed to load identification model — {e}")

    def identify(self, frame: np.ndarray) -> List[Dict]:
        """Identify faces in a frame.
        
        Returns:
            List of dicts with keys: id, name, distance (lower is better).
        """
        # Quick check if we have any faces to match against
        # This is faster than iterating through the database
        if not any(self._db_path.iterdir()):
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
            if df.empty:
                continue

            # Get the best match (first row)
            match = df.iloc[0]
            identity_path = Path(match["identity"])
            
            # The person_id is the parent directory name
            # Structure: data/faces/{person_id}/image.jpg
            person_id = identity_path.parent.name
            
            name = self.people.get_name(person_id)
            if not name:
                continue

            self.people.touch(person_id)
            
            results.append({
                "id": person_id,
                "name": name,
                # DeepFace.find returns 'distance' (dissimilarity), not confidence.
                # Lower distance = higher confidence.
                "distance": match.get("distance", 0.0),
            })

        return results
