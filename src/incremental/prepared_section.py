from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.state_store import ChunkRecord as StateChunkRecord


@dataclass(frozen=True)
class PreparedSection:
    order_in_parent: int
    heading: str
    level: int
    section_hash: str
    chunk_records: List[StateChunkRecord]
