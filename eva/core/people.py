import sqlite3
from datetime import datetime, timezone
from typing import Dict
from config import logger, DATA_DIR


class PeopleDB:
    """EVA's memory of people she's met."""

    def __init__(self):
        self._cache = None
        self.init_db()
        logger.debug(f"PeopleDB: {len(self._cache)} people in memory.")

    def _connect(self) -> sqlite3.Connection:
        """Connect to the database."""
        db_path = DATA_DIR / "database" / "eva.db"
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        """Initialize the database."""
        (DATA_DIR / "database").mkdir(parents=True, exist_ok=True)
        self._create_table()
        self._cache = self._load_all()

    def _create_table(self) -> None:
        """Create the people table if it doesn't exist."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS people (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    relationship TEXT,
                    first_seen TIMESTAMP,
                    last_seen TIMESTAMP,
                    notes TEXT
                )
            """)

    def _load_all(self) -> Dict[str, Dict]:
        """Load all people from the database."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM people").fetchall()
        return {row["id"]: dict(row) for row in rows}

    def get(self, person_id: str) -> Dict | None:
        """Get a person from the database."""
        return self._cache.get(person_id)

    def get_name(self, person_id: str) -> str | None:
        """Get the name of a person from the database."""
        person = self._cache.get(person_id)
        return person["name"] if person else None

    def get_all(self) -> Dict[str, Dict]:
        """Get all people from the database."""
        return self._cache

    def add(self, person_id: str, name: str, relationship: str = None) -> bool:
        """Register a new person to the database."""
        if person_id in self._cache:
            logger.warning(f"PeopleDB: {person_id} already exists.")
            return False

        now = datetime.now(timezone.utc).isoformat()
        face_dir = DATA_DIR / "faces" / person_id
        face_dir.mkdir(parents=True, exist_ok=True)

        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO people (id, name, relationship, first_seen, last_seen) VALUES (?, ?, ?, ?, ?)",
                    (person_id, name, relationship, now, now),
                )
            self._cache[person_id] = {
                "id": person_id, "name": name, "relationship": relationship,
                "first_seen": now, "last_seen": now, "notes": None,
            }
            logger.info(f"PeopleDB: Added {name} ({person_id}).")
            return True
        except sqlite3.Error as e:
            logger.error(f"PeopleDB: Failed to add {person_id} — {e}")
            return False

    def touch(self, person_id: str) -> None:
        """Update last_seen to now."""
        now = datetime.now(timezone.utc).isoformat()
        try:
            with self._connect() as conn:
                conn.execute("UPDATE people SET last_seen = ? WHERE id = ?", (now, person_id))
            if person_id in self._cache:
                self._cache[person_id]["last_seen"] = now
        except sqlite3.Error as e:
            logger.error(f"PeopleDB: Failed to touch {person_id} — {e}")

    def append_notes(self, person_id: str, impression: str) -> None:
        """EVA adds a new impression, timestamped for future consolidation."""
        if person_id not in self._cache:
            return

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        entry = f"## {timestamp}\n\n{impression.strip()}"
        existing = self._cache[person_id].get("notes") or ""
        updated = f"{existing}\n\n{entry}".strip() if existing else entry

        try:
            with self._connect() as conn:
                conn.execute("UPDATE people SET notes = ? WHERE id = ?", (updated, person_id))
            self._cache[person_id]["notes"] = updated
            logger.debug(f"PeopleDB: noted impression for {person_id}.")
        except sqlite3.Error as e:
            logger.error(f"PeopleDB: Failed to update notes for {person_id} — {e}")
