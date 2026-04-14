from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


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
