from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DocumentRecord:
    path: str
    last_modified_at: float
    size: int
    doc_hash: Optional[str]
    is_active: bool = True
