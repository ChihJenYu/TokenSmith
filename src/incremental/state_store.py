from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class DocumentRecord:
    path: str
    last_modified_at: float
    size: int
    doc_hash: Optional[str]
    is_active: bool = True


@dataclass(frozen=True)
class SectionRecord:
    document_id: int
    order_in_parent: int
    heading: str
    level: int
    section_hash: Optional[str]
    is_active: bool = True


@dataclass(frozen=True)
class ChunkRecord:
    section_id: int
    order_in_parent: int
    chunk_hash: str
    text: str
    bm25_tokens: Optional[Any] = None
    embeddding: Optional[bytes] = None
    metadata_json: Optional[Any] = None
    is_active: bool = True


class StateStore:
    def __init__(self, db_path: os.PathLike | str) -> None:
        self.db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None

    @property
    def connection(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("StateStore is not connected. Call connect() first.")
        return self._conn

    def connect(self) -> sqlite3.Connection:
        if self._conn is None:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            self._conn = conn
            self._create_or_load_schema()
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def get_document_by_path(self, path: str) -> Optional[sqlite3.Row]:
        return self.connection.execute(
            "SELECT * FROM documents WHERE path = ?",
            (path,),
        ).fetchone()

    def get_sections_for_document(
        self,
        document_id: int,
        *,
        is_active: bool = False,
    ) -> list[sqlite3.Row]:
        query = "SELECT * FROM sections WHERE document_id = ?"
        params: list[Any] = [document_id]
        if is_active:
            query += " AND is_active = 1"
        query += " ORDER BY order_in_parent"
        return list(self.connection.execute(query, params).fetchall())

    def get_chunks_for_section(
        self,
        section_id: int,
        *,
        is_active: bool = False,
    ) -> list[sqlite3.Row]:
        query = "SELECT * FROM chunks WHERE section_id = ?"
        params: list[Any] = [section_id]
        if is_active:
            query += " AND is_active = 1"
        query += " ORDER BY order_in_parent"
        return list(self.connection.execute(query, params).fetchall())

    def get_chunk_by_hash(self, chunk_hash: str) -> Optional[sqlite3.Row]:
        return self.connection.execute(
            """
            SELECT *
            FROM chunks
            WHERE chunk_hash = ?
            ORDER BY is_active DESC, updated_at DESC
            LIMIT 1
            """,
            (chunk_hash,),
        ).fetchone()

    def get_all_active_chunks(self) -> Iterable[sqlite3.Row]:
        return self.connection.execute(
            """
            SELECT c.*, s.document_id
            FROM chunks c
            JOIN sections s ON s.id = c.section_id
            WHERE c.is_active = 1 AND s.is_active = 1
            ORDER BY s.document_id, s.order_in_parent, c.order_in_parent
            """
        )

    def upsert_document(self, record: DocumentRecord) -> int:
        cursor = self.connection.execute(
            """
            INSERT INTO documents (path, last_modified_at, size, doc_hash, is_active)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                last_modified_at = excluded.last_modified_at,
                size = excluded.size,
                doc_hash = excluded.doc_hash,
                is_active = excluded.is_active,
                updated_at = CURRENT_TIMESTAMP
            RETURNING id
            """,
            (
                record.path,
                record.last_modified_at,
                record.size,
                record.doc_hash,
                int(record.is_active),
            ),
        )
        row = cursor.fetchone()
        self.connection.commit()
        return int(row["id"])

    def upsert_section(self, record: SectionRecord) -> int:
        cursor = self.connection.execute(
            """
            INSERT INTO sections (document_id, order_in_parent, heading, level, section_hash, is_active)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(document_id, order_in_parent) DO UPDATE SET
                heading = excluded.heading,
                level = excluded.level,
                section_hash = excluded.section_hash,
                is_active = excluded.is_active,
                updated_at = CURRENT_TIMESTAMP
            RETURNING id
            """,
            (
                record.document_id,
                record.order_in_parent,
                record.heading,
                record.level,
                record.section_hash,
                int(record.is_active),
            ),
        )
        row = cursor.fetchone()
        self.connection.commit()
        return int(row["id"])

    def upsert_chunk(self, record: ChunkRecord) -> int:
        cursor = self.connection.execute(
            """
            INSERT INTO chunks (
                section_id,
                order_in_parent,
                chunk_hash,
                text,
                bm25_tokens,
                embeddding,
                metadata_json,
                is_active
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(section_id, order_in_parent) DO UPDATE SET
                chunk_hash = excluded.chunk_hash,
                text = excluded.text,
                bm25_tokens = excluded.bm25_tokens,
                embeddding = excluded.embeddding,
                metadata_json = excluded.metadata_json,
                is_active = excluded.is_active,
                updated_at = CURRENT_TIMESTAMP
            RETURNING id
            """,
            (
                record.section_id,
                record.order_in_parent,
                record.chunk_hash,
                record.text,
                self._serialize_json(record.bm25_tokens),
                record.embeddding,
                self._serialize_json(record.metadata_json),
                int(record.is_active),
            ),
        )
        row = cursor.fetchone()
        self.connection.commit()
        return int(row["id"])

    def mark_document_inactive(self, path: str) -> None:
        self.connection.execute(
            "UPDATE documents SET is_active = 0, updated_at = CURRENT_TIMESTAMP WHERE path = ?",
            (path,),
        )
        self.connection.commit()

    def mark_sections_inactive(self, document_id: int) -> None:
        self.connection.execute(
            """
            UPDATE sections
            SET is_active = 0, updated_at = CURRENT_TIMESTAMP
            WHERE document_id = ?
            """,
            (document_id,),
        )
        self.connection.commit()

    def mark_chunks_inactive(self, section_id: int) -> None:
        self.connection.execute(
            """
            UPDATE chunks
            SET is_active = 0, updated_at = CURRENT_TIMESTAMP
            WHERE section_id = ?
            """,
            (section_id,),
        )
        self.connection.commit()

    def _create_or_load_schema(self) -> None:
        """
        Ensure the expected tables and indexes are available on this connection.
        """
        self._create_tables_if_missing()
        self._create_indexes_if_missing()
        self.connection.commit()

    def _create_tables_if_missing(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL UNIQUE,
                last_modified_at REAL NOT NULL,
                size INTEGER NOT NULL,
                doc_hash TEXT,
                is_active INTEGER NOT NULL DEFAULT 1,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS sections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                order_in_parent INTEGER NOT NULL,
                heading TEXT NOT NULL,
                level INTEGER NOT NULL,
                section_hash TEXT,
                is_active INTEGER NOT NULL DEFAULT 1,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE,
                UNIQUE(document_id, order_in_parent)
            )
            """
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                section_id INTEGER NOT NULL,
                order_in_parent INTEGER NOT NULL,
                chunk_hash TEXT NOT NULL,
                text TEXT NOT NULL,
                bm25_tokens TEXT,
                embeddding BLOB,
                metadata_json TEXT,
                is_active INTEGER NOT NULL DEFAULT 1,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(section_id) REFERENCES sections(id) ON DELETE CASCADE,
                UNIQUE(section_id, order_in_parent)
            )
            """
        )

    def _create_indexes_if_missing(self) -> None:
        statements = [
            "CREATE INDEX IF NOT EXISTS idx_documents_path ON documents(path)",
            "CREATE INDEX IF NOT EXISTS idx_sections_document_id ON sections(document_id)",
            "CREATE INDEX IF NOT EXISTS idx_chunks_section_id ON chunks(section_id)",
            "CREATE INDEX IF NOT EXISTS idx_chunks_chunk_hash ON chunks(chunk_hash)",
        ]
        for statement in statements:
            self.connection.execute(statement)

    @staticmethod
    def _serialize_json(value: Optional[Any]) -> Optional[str]:
        if value is None:
            return None
        return json.dumps(value, sort_keys=True)
