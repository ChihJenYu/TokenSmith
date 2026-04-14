from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Optional

from .chunk_record import ChunkRecord
from .document_record import DocumentRecord
from .section_record import SectionRecord


class StateStore:
    def __init__(self, db_path: os.PathLike | str) -> None:
        self.db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None
        self._schema_path = Path(__file__).with_name("schema.sql")

    @property
    def connection(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("StateStore is not connected. Call connect() first.")
        return self._conn

    def connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
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

    def get_all_documents(self, *, is_active: bool = False) -> list[sqlite3.Row]:
        query = "SELECT * FROM documents"
        if is_active:
            query += " WHERE is_active = 1"
        query += " ORDER BY path"
        return list(self.connection.execute(query).fetchall())

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
            SELECT c.*, s.document_id, d.path AS source_path
            FROM chunks c
            JOIN sections s ON s.id = c.section_id
            JOIN documents d ON d.id = s.document_id
            WHERE c.is_active = 1 AND s.is_active = 1 AND d.is_active = 1
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
        self.connection.executescript(self._schema_path.read_text(encoding="utf-8"))
        self.connection.commit()

    @staticmethod
    def _serialize_json(value: Optional[Any]) -> Optional[str]:
        if value is None:
            return None
        return json.dumps(value, sort_keys=True)
