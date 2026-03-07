"""Tests for MemoryDB, JournalDB, PeopleDB — memory pipeline."""

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from eva.agent.schema import PeopleReflection, PersonImpression
from eva.utils.prompt import load_prompt
import eva.core.db as db_module
import eva.core.people as people_module


def _patch_data_dir(tmp_path):
    """Point DATA_DIR at a temp directory."""
    db_module.DATA_DIR = tmp_path
    people_module.DATA_DIR = tmp_path


async def make_memory(tmp_path):
    """Create a MemoryDB with mocked LLM pointing at a temp database."""
    _patch_data_dir(tmp_path)

    from eva.core.db import SQLiteHandler
    from eva.core.journal import JournalDB
    from eva.core.people import PeopleDB
    from eva.core.memory import MemoryDB

    handler = SQLiteHandler()
    people_db = PeopleDB(handler)
    journal_db = JournalDB(handler)
    await asyncio.gather(people_db.init_db(), journal_db.init_db())

    # Build MemoryDB without hitting a real LLM
    mem = MemoryDB.__new__(MemoryDB)
    mem._journal = journal_db
    mem._people = people_db
    mem._pen = None
    mem._session_people_ids = set()
    mem._journal_prompt = load_prompt("journal")
    mem._relationships_prompt = load_prompt("relationships")

    return mem, handler


async def make_people_db(tmp_path):
    """Create a PeopleDB pointing at a temp database."""
    _patch_data_dir(tmp_path)

    from eva.core.db import SQLiteHandler
    from eva.core.people import PeopleDB

    handler = SQLiteHandler()
    pdb = PeopleDB(handler)
    await pdb.init_db()
    return pdb, handler


def test_journal_write_and_read():
    async def _run():
        with tempfile.TemporaryDirectory() as tmp:
            db, handler = await make_memory(Path(tmp))
            entry_id = await db._journal.add("I talked with Alice about cats.", session_id="s1")
            assert entry_id
            recent = await db._journal.get_recent()
            assert len(recent) == 1 and "Alice" in recent[0]
            await handler.close_all()
    asyncio.run(_run())


def test_journal_recency():
    async def _run():
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db, handler = await make_memory(tmp_path)

            import sqlite3
            conn = sqlite3.connect(tmp_path / "database" / "eva.db")
            conn.execute(
                "INSERT INTO journal (id, content, session_id, created_at) VALUES (?, ?, ?, ?)",
                ("old1", "Yesterday's memory", "s0", "2020-01-01T00:00:00+00:00"),
            )
            conn.commit()
            conn.close()

            recent = await db._journal.get_recent()
            assert len(recent) == 1 and "Yesterday" in recent[0]
            await handler.close_all()
    asyncio.run(_run())


def test_distill_collapses_feel_speak():
    from eva.core.memory import MemoryDB

    messages = [
        HumanMessage(content="I hear: hello"),
        AIMessage(content="", tool_calls=[
            {"id": "tc1", "name": "feel", "args": {"feeling": "curious", "inner_monologue": "someone said hi"}},
            {"id": "tc2", "name": "speak", "args": {"text": "Hi there!"}},
        ]),
        ToolMessage(content="ok", tool_call_id="tc1"),
        ToolMessage(content="ok", tool_call_id="tc2"),
        HumanMessage(content="I hear: how are you?"),
        AIMessage(content="", tool_calls=[
            {"id": "tc3", "name": "feel", "args": {"feeling": "happy", "inner_monologue": "they care"}},
        ]),
    ]

    distilled = MemoryDB.distill(messages)
    assert isinstance(distilled[0], HumanMessage)
    assert isinstance(distilled[1], AIMessage)
    assert "[I felt curious]" in distilled[1].content
    assert 'I said: "Hi there!"' in distilled[1].content
    assert isinstance(distilled[2], HumanMessage)
    assert "how are you" in distilled[2].content
    assert isinstance(distilled[3], AIMessage)
    assert distilled[3].tool_calls


def test_distill_preserves_non_tool_ai():
    from eva.core.memory import MemoryDB

    messages = [
        HumanMessage(content="I see: empty room"),
        AIMessage(content="Nothing interesting here."),
        HumanMessage(content="I hear: test"),
    ]

    distilled = MemoryDB.distill(messages)
    assert len(distilled) == 3
    assert distilled[1].content == "Nothing interesting here."


def test_flush_no_llm():
    async def _run():
        with tempfile.TemporaryDirectory() as tmp:
            db, handler = await make_memory(Path(tmp))
            messages = [
                HumanMessage(content="I hear: good morning"),
                AIMessage(content="", tool_calls=[
                    {"id": "tc1", "name": "speak", "args": {"text": "Good morning!"}},
                ]),
                ToolMessage(content="ok", tool_call_id="tc1"),
            ]
            result = await db.flush(messages, session_id="test-session")
            assert result is True
            recent = await db._journal.get_recent()
            assert len(recent) == 1 and "morning" in recent[0].lower()
            await handler.close_all()
    asyncio.run(_run())


def test_flush_with_llm():
    async def _run():
        with tempfile.TemporaryDirectory() as tmp:
            db, handler = await make_memory(Path(tmp))

            mock_response = MagicMock()
            mock_response.content = "I greeted someone good morning and they responded warmly."
            mock_llm = AsyncMock()
            mock_llm.ainvoke.return_value = mock_response
            db._pen = mock_llm

            messages = [
                HumanMessage(content="I hear: good morning"),
                AIMessage(content="", tool_calls=[
                    {"id": "tc1", "name": "speak", "args": {"text": "Good morning!"}},
                ]),
                ToolMessage(content="ok", tool_call_id="tc1"),
            ]
            result = await db.flush(messages, session_id="test-session")
            assert result is True
            assert mock_llm.ainvoke.called
            recent = await db._journal.get_recent()
            assert len(recent) == 1 and "greeted" in recent[0]
            await handler.close_all()
    asyncio.run(_run())


def test_flush_empty():
    async def _run():
        with tempfile.TemporaryDirectory() as tmp:
            db, handler = await make_memory(Path(tmp))
            result = await db.flush([], session_id="empty")
            assert result is False
            await handler.close_all()
    asyncio.run(_run())


def test_prepare_context():
    async def _run():
        with tempfile.TemporaryDirectory() as tmp:
            db, handler = await make_memory(Path(tmp))
            await db._journal.add("Earlier I greeted Alice.", session_id="s1")
            distilled, journal = await db.prepare_context([HumanMessage(content="I hear: hello again")])
            assert len(distilled) == 1 and "Alice" in journal
            await handler.close_all()
    asyncio.run(_run())


def test_append_notes():
    async def _run():
        with tempfile.TemporaryDirectory() as tmp:
            pdb, handler = await make_people_db(Path(tmp))
            await pdb.add("alice", "Alice Chen", "friend")
            await pdb.append_notes("alice", "Seemed tired but was patient.")
            await pdb.append_notes("alice", "Brought coffee, was in a great mood.")
            person = pdb.get("alice")
            assert person is not None
            notes = person["notes"]
            assert "Seemed tired" in notes and "Brought coffee" in notes
            assert notes.count("## ") == 2
            await handler.close_all()
    asyncio.run(_run())


def test_reflect_people():
    async def _run():
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pdb, handler = await make_people_db(tmp_path)
            await pdb.add("alice", "Alice Chen", "friend")
            await pdb.add("bob", "Bob", "colleague")

            db, _ = await make_memory(tmp_path)
            db._people = pdb

            # Track people in session
            db.add_people_to_session({"alice", "bob"})

            mock_reflection = PeopleReflection(impressions=[
                PersonImpression(person_id="alice", impression="She was excited about the new project."),
                PersonImpression(person_id="bob", impression="Quiet today, just listened."),
            ])
            mock_structured = AsyncMock()
            mock_structured.ainvoke.return_value = mock_reflection
            mock_llm = MagicMock()
            mock_llm.with_structured_output.return_value = mock_structured
            db._pen = mock_llm

            await db._reflect_people("Alice talked about her project. Bob was quiet.")

            alice = pdb.get("alice")
            assert alice is not None and "excited" in alice["notes"]
            bob = pdb.get("bob")
            assert bob is not None and "Quiet" in bob["notes"]
            await handler.close_all()
    asyncio.run(_run())


def test_reflect_skips_without_people():
    async def _run():
        with tempfile.TemporaryDirectory() as tmp:
            db, handler = await make_memory(Path(tmp))
            await db._reflect_people("Some conversation text")
            await handler.close_all()
    asyncio.run(_run())


def test_no_circular_import():
    """Brain should be importable without circular dependency."""
    from eva.core.graph import Brain
    assert Brain is not None
