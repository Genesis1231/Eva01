"""Shared async SQLite handler for EVA databases."""

import asyncio
from typing import Any, Iterable

import aiosqlite

from config import DATA_DIR, SQLITE_DB_NAME, logger


class SQLiteHandler:
    """Centralized connection and query helper for async SQLite access."""

    def __init__(self):
        self._connections: dict[str, aiosqlite.Connection] = {}
        self._write_locks: dict[str, asyncio.Lock] = {}
        self._init_lock = asyncio.Lock()
        self._db_dir = DATA_DIR / "database"
        self._vec_extension_loaded = False

    @property
    def vec_extension_loaded(self) -> bool:
        """Whether sqlite-vec extension was loaded on at least one connection."""
        return self._vec_extension_loaded

    async def _try_load_vec_extension(self, conn: aiosqlite.Connection) -> None:
        """Try loading sqlite-vec extension without failing startup."""
        try:
            import sqlite_vec  # type: ignore

            sqlite_vec.load(conn._conn)  # aiosqlite wraps sqlite3 connection
            self._vec_extension_loaded = True
            logger.debug("SQLiteHandler: sqlite-vec extension loaded.")
        except Exception as e:
            logger.debug(f"SQLiteHandler: sqlite-vec not loaded ({e}).")

    async def _get_connection(self, db_name: str = SQLITE_DB_NAME) -> aiosqlite.Connection:
        """Get or create a connection for the specified database name."""
        conn = self._connections.get(db_name)
        if conn is not None:
            return conn

        # Locked initialization to prevent race conditions during startup.
        async with self._init_lock:
            if db_name in self._connections:
                return self._connections[db_name]

            self._db_dir.mkdir(parents=True, exist_ok=True)
            db_path = self._db_dir / db_name
            conn = await aiosqlite.connect(db_path, check_same_thread=False)
            conn.row_factory = aiosqlite.Row
            await self._try_load_vec_extension(conn)
            await conn.execute("PRAGMA journal_mode=WAL;")
            await conn.execute("PRAGMA busy_timeout=5000;")
            await conn.execute("PRAGMA foreign_keys=ON;")
            await conn.commit()

            self._connections[db_name] = conn
            self._write_locks[db_name] = asyncio.Lock()
            return conn

    async def execute(
        self,
        query: str,
        params: tuple[Any, ...] = (),
        db_name: str = SQLITE_DB_NAME,
    ) -> None:
        """Execute a query that modifies the database."""
        conn = await self._get_connection(db_name)
        async with self._write_locks[db_name]:
            await conn.execute(query, params)
            await conn.commit()

    async def executemany(
        self,
        query: str,
        params_list: list[tuple[Any, ...]],
        db_name: str = SQLITE_DB_NAME,
    ) -> None:
        """Execute a query with multiple sets of parameters, atomically."""
        conn = await self._get_connection(db_name)
        async with self._write_locks[db_name]:
            try:
                await conn.executemany(query, params_list)
                await conn.commit()
            except BaseException:
                # Roll back rows already applied — otherwise they linger uncommitted
                # on the shared connection and the next unrelated commit persists them.
                await conn.rollback()
                raise

    async def execute_transaction(
        self,
        statements: list[tuple[str, tuple[Any, ...]]],
        db_name: str = SQLITE_DB_NAME,
    ) -> None:
        """Execute several write statements as one atomic transaction (all-or-nothing)."""
        conn = await self._get_connection(db_name)
        async with self._write_locks[db_name]:
            try:
                for query, params in statements:
                    await conn.execute(query, params)
                await conn.commit()
            except BaseException:
                await conn.rollback()
                raise

    async def execute_insert(
        self,
        query: str,
        params: tuple[Any, ...] = (),
        db_name: str = SQLITE_DB_NAME,
    ) -> int:
        """Execute an INSERT query and return SQLite lastrowid atomically."""
        conn = await self._get_connection(db_name)
        async with self._write_locks[db_name]:
            cursor = await conn.execute(query, params)
            await conn.commit()
            if cursor.lastrowid is None:
                raise RuntimeError("SQLite insert did not return lastrowid")
            return int(cursor.lastrowid)

    async def fetchall(
        self,
        query: str,
        params: tuple[Any, ...] = (),
        db_name: str = SQLITE_DB_NAME,
    ) -> Iterable[aiosqlite.Row]:
        """Execute a SELECT query and return all results."""
        conn = await self._get_connection(db_name)
        async with conn.execute(query, params) as cursor:
            return await cursor.fetchall()

    async def fetchone(
        self,
        query: str,
        params: tuple[Any, ...] = (),
        db_name: str = SQLITE_DB_NAME,
    ) -> aiosqlite.Row | None:
        """Execute a SELECT query and return a single result."""
        conn = await self._get_connection(db_name)
        async with conn.execute(query, params) as cursor:
            return await cursor.fetchone()

    async def close(self, db_name: str = SQLITE_DB_NAME) -> None:
        """Close the connection for the specified database name."""
        async with self._init_lock:
            conn = self._connections.pop(db_name, None)
            self._write_locks.pop(db_name, None)
            if conn is not None:
                await conn.close()

    async def close_all(self) -> None:
        """Close all database connections."""
        async with self._init_lock:
            for conn in self._connections.values():
                await conn.close()
            self._connections.clear()
            self._write_locks.clear()
