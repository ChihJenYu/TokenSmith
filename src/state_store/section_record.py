from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SectionRecord:
    document_id: int
    order_in_parent: int
    heading: str
    level: int
    section_hash: Optional[str]
    is_active: bool = True
