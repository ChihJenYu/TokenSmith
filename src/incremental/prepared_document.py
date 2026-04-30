from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .prepared_section import PreparedSection


@dataclass(frozen=True)
class PreparedDocument:
    path: str
    last_modified_at: float
    size: int
    doc_hash: str
    sections: List[PreparedSection]
