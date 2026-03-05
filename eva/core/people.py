import sqlite3
from datetime import datetime, timezone
from typing import Dict, List
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


    def append_notes(
        self,
        person_id: str = None,
        impression: str = None,
        *,
        raw: str = None,
    ) -> None:
        """
        Append impressions of people. 
        Single: (person_id, impression). 
        Multi-line: (raw=...) with 'id: impression' lines.
        """
        timestamp = datetime.now(timezone.utc).strftime("%B %d, %Y")
        updates: List[tuple] = []

        if raw:
            all_people = self.get_all()
            for line in raw.splitlines():
                line = line.strip()
                if not line or ":" not in line:
                    continue
                pid, _, imp = line.partition(":")
                pid, imp = pid.strip(), imp.strip()
                if pid in all_people and imp:
                    updates.append((pid, imp))
        
        elif person_id and impression:
            if person_id in self._cache:
                updates.append((person_id, impression.strip()))

        if not updates:
            return

        try:
            with self._connect() as conn:
                for pid, imp in updates:
                    entry = f"## {timestamp}\n\n{imp}\n\n"
                    existing = self._cache.get(pid, {}).get("notes") or ""
                    updated = f"{existing}\n\n{entry}".strip() if existing else entry
                    conn.execute("UPDATE people SET notes = ? WHERE id = ?", (updated, pid))
                    if pid in self._cache:
                        self._cache[pid]["notes"] = updated
            for pid, _ in updates:
                logger.debug(f"PeopleDB: noted impression for {pid}.")
        except sqlite3.Error as e:
            logger.error(f"PeopleDB: Failed to update notes — {e}")

