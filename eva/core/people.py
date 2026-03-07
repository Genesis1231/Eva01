""" 
EVA's memory of people she's met, 
stored in a SQLite database with an in-memory cache for quick access. 
"""

from datetime import datetime, timezone
from typing import Iterable, Dict, Set

from config import logger, DATA_DIR
from eva.core.db import SQLiteHandler


class PeopleDB:
    """EVA's memory of people she's met."""

    def __init__(self, db: SQLiteHandler):
        self._db = db
        self._cache = {}
        self._initialized = False

    async def init_db(self) -> None:
        """Initialize the database."""
        if self._initialized:
            return

        await self._create_table()
        self._cache = await self._load_all()
        self._initialized = True
        logger.debug(f"PeopleDB: {len(self._cache)} people in memory.")

    async def _create_table(self) -> None:
        """Create the people table if it doesn't exist."""
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS people (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                relationship TEXT,
                first_seen TIMESTAMP,
                last_seen TIMESTAMP,
                notes TEXT
            )
            """
        )

    async def _load_all(self) -> Dict[str, Dict]:
        """Load all people from the database."""
        rows = await self._db.fetchall("SELECT * FROM people")
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

    def get_many(self, person_ids: Iterable[str]) -> Dict[str, Dict]:
        """Get a subset of known people by id."""
        return {
            person_id: person
            for person_id in person_ids
            if (person := self._cache.get(person_id)) is not None
        }

    def get_id_name_map(self) -> Dict[str, str]:
        """Get a lightweight snapshot of known people: id -> name."""
        return {
            person_id: person["name"]
            for person_id, person in self._cache.items()
            if person.get("name")
        }


    async def add(self, person_id: str, name: str, relationship: str | None = None) -> bool:
        """Register a new person to the database."""
        if person_id in self._cache:
            logger.warning(f"PeopleDB: {person_id} already exists.")
            return False

        now = datetime.now(timezone.utc).isoformat()
        face_dir = DATA_DIR / "faces" / person_id
        face_dir.mkdir(parents=True, exist_ok=True)

        try:
            await self._db.execute(
                "INSERT INTO people (id, name, relationship, first_seen, last_seen) VALUES (?, ?, ?, ?, ?)",
                (person_id, name, relationship, now, now),
            )
            self._cache[person_id] = {
                "id": person_id, "name": name, "relationship": relationship,
                "first_seen": now, "last_seen": now, "notes": None,
            }
            logger.debug(f"PeopleDB: Added {name} ({person_id}).")
            return True
        except Exception as e:
            logger.error(f"PeopleDB: Failed to add {person_id} — {e}")
            return False

    async def touch(self, person_id: str | Set[str]) -> None:
        """Update last_seen to now for one or multiple people."""

        if not person_id:
            return
                    
        now = datetime.now(timezone.utc).isoformat()
        try:
            if isinstance(person_id, str):
                person_ids = {person_id}
            else:
                person_ids = set(person_id)

            await self._db.executemany(
                "UPDATE people SET last_seen = ? WHERE id = ?",
                [(now, pid) for pid in person_ids],
            )

            for pid in person_ids:
                if pid in self._cache:
                    self._cache[pid]["last_seen"] = now

            logger.debug(f"PeopleDB: Touched {', '.join(person_ids)}.")
                
        except Exception as e:
            logger.error(f"PeopleDB: Failed to touch {person_id} — {e}")

    async def append_notes(self, person_id: str, impression: str) -> None:
        """EVA adds a new impression, timestamped for future consolidation."""
        if person_id not in self._cache:
            return

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        entry = f"## {timestamp}\n\n{impression.strip()}"
        existing = self._cache[person_id].get("notes") or ""
        updated = f"{existing}\n\n{entry}".strip() if existing else entry

        try:
            await self._db.execute(
                "UPDATE people SET notes = ? WHERE id = ?",
                (updated, person_id)
            )
            self._cache[person_id]["notes"] = updated
            logger.debug(f"PeopleDB: noted impression for {person_id}.")
        except Exception as e:
            logger.error(f"PeopleDB: Failed to update notes for {person_id} — {e}")

    async def append_reflection_notes(self, mentioned: set[str], impressions: list) -> None:
        """Persist validated LLM impressions for people that were in scope."""
        for entry in impressions:
            if entry.person_id in mentioned:
                await self.append_notes(entry.person_id, entry.impression)

    @staticmethod
    def render_people(people: Dict[str, Dict]) -> str:
        """Render people context lines for the relationship reflection prompt."""
        lines = []
        for pid, person in people.items():
            rel = person.get("relationship") or "no relationship noted"
            lines.append(f"{pid}: {person['name']} ({rel})")
        return "\n".join(lines)