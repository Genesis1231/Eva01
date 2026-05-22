"""Interactive ID manager for EVA face/voice enrollment.

Usage:
    python idconfig.py

Features:
- Shows current ID table: ID / Name / Face ID / Voice ID.
- Register new ID (auto person_id generation).
- Delete ID from database (face/voice folders are preserved).
"""

import asyncio
import re
from pathlib import Path

from tabulate import tabulate

from config import DATA_DIR
from eva.database.db import SQLiteHandler
from eva.core.people import PeopleDB
from config.record_void import PROMPTS, SAMPLE_RATE, record_one
from eva.senses.audio.vad_onnx import SileroVAD
import soundfile as sf

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _count_face_images(person_id: str) -> int:
    face_dir = DATA_DIR / "faces" / person_id
    if not face_dir.exists():
        return 0
    return sum(
        1
        for p in face_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
    )


def _count_voice_samples(person_id: str) -> int:
    voice_dir = DATA_DIR / "voices" / person_id
    if not voice_dir.exists():
        return 0
    return sum(1 for p in voice_dir.glob("*.wav") if p.is_file())


def _next_person_id(existing_ids: list[str]) -> str:
    used = set()
    for person_id in existing_ids:
        match = re.fullmatch(r"p(\d+)", person_id)
        if match:
            used.add(int(match.group(1)))

    n = 1
    while n in used:
        n += 1
    return f"p{n:03d}"


def _clear_voice_cache() -> None:
    cache = DATA_DIR / "voices" / ".embeddings_cache.pkl"
    if cache.exists():
        cache.unlink()
        print("Voice embeddings cache cleared.")


def _show_table(people: dict[str, dict]) -> None:
    rows = []
    for person_id in sorted(people):
        person = people[person_id]
        rows.append(
            [
                person_id,
                person.get("name", ""),
                _count_face_images(person_id),
                _count_voice_samples(person_id),
            ]
        )

    print("\nCurrent Identities:")
    if not rows:
        print("(empty)")
        return

    table_text = tabulate(
        rows,
        headers=["ID", "Name", "Face ID", "Voice ID"],
        tablefmt="github",
    )
    table_width = max(len(line) for line in table_text.splitlines())
    border = "-" * table_width

    print(border)
    print(table_text)
    print(border)


def _record_voice_samples(person_id: str) -> int:
    out_dir = DATA_DIR / "voices" / person_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nLoading Silero VAD...")
    vad = SileroVAD()

    print(f"Recording 5 voice samples for '{person_id}'")
    print(f"Saving to: {out_dir}")

    # Replace existing samples for deterministic enrollment.
    for wav in out_dir.glob("*.wav"):
        wav.unlink()

    saved = 0
    for i, prompt in enumerate(PROMPTS):
        audio = record_one(i, prompt, vad)
        if audio is None:
            retry = input("  Retry this sample? [Y/n] ").strip().lower()
            if retry != "n":
                audio = record_one(i, prompt, vad)

        if audio is not None:
            path = out_dir / f"sample_{i + 1:02d}.wav"
            sf.write(str(path), audio, SAMPLE_RATE)
            print(f"  Saved: {path.name}")
            saved += 1

    if saved > 0:
        _clear_voice_cache()

    print(f"Done: {saved}/5 voice samples saved.")
    return saved


async def _register(people_db: PeopleDB) -> None:
    people = people_db.get_all()
    person_id = _next_person_id(list(people.keys()))

    print(f"\nRegistering new ID: {person_id}")
    while True:
        name = input("Enter name: ").strip()
        if name:
            break
        print("Name cannot be empty.")

    created = await people_db.add(person_id, name)
    if not created:
        print("Failed to create ID in database.")
        return

    face_dir = DATA_DIR / "faces" / person_id
    voice_dir = DATA_DIR / "voices" / person_id
    face_dir.mkdir(parents=True, exist_ok=True)
    voice_dir.mkdir(parents=True, exist_ok=True)

    print(f"Face folder:  {face_dir}")
    print(f"Voice folder: {voice_dir}")
    print("Add face images to the face folder now.")

    while True:
        face_count = _count_face_images(person_id)
        if face_count > 0:
            print(f"Detected {face_count} face image(s) (Recommend: 3+ front photos).")
            break

        choice = input("No face images detected yet. \n[R]echeck or [S]kip for now? ").strip().lower()
        if choice == "s":
            break

    _record_voice_samples(person_id)


async def _reload_people(db: SQLiteHandler) -> PeopleDB:
    people_db = PeopleDB(db)
    await people_db.init_db()
    return people_db


async def _delete(people_db: PeopleDB, db: SQLiteHandler) -> PeopleDB:
    people = people_db.get_all()
    ids = sorted(people.keys())

    if not ids:
        print("\nNo IDs to delete.")
        return people_db

    print("\nSelect ID to delete:")
    for i, person_id in enumerate(ids, start=1):
        print(f"{i}. {person_id} ({people[person_id].get('name', '')})")

    raw = input("Choice: ").strip()
    if not raw.isdigit() or not (1 <= int(raw) <= len(ids)):
        print("Invalid selection.")
        return people_db

    person_id = ids[int(raw) - 1]
    confirm = input(f"Delete '{person_id}' from DB only? [y/N] ").strip().lower()
    if confirm != "y":
        print("Deletion cancelled.")
        return people_db

    await db.execute("DELETE FROM people WHERE id = ?", (person_id,))
    print(f"Deleted {person_id} from database.")
    print(f"Preserved folders: {DATA_DIR / 'faces' / person_id} and {DATA_DIR / 'voices' / person_id}")
    return await _reload_people(db)


async def run_cli() -> int:
    db = SQLiteHandler()
    people_db = await _reload_people(db)

    try:
        while True:
            _show_table(people_db.get_all())
            print("\nMain Menu")
            print("1. Register new ID")
            print("2. Delete ID")
            print("3. Exit")

            choice = input("Select: ").strip()
            if choice == "1":
                await _register(people_db)
                people_db = await _reload_people(db)
            elif choice == "2":
                people_db = await _delete(people_db, db)
            elif choice == "3":
                print("Bye.")
                return 0
            else:
                print("Invalid choice.")
    finally:
        await db.close_all()


def main() -> int:
    return asyncio.run(run_cli())


if __name__ == "__main__":
    raise SystemExit(main())
