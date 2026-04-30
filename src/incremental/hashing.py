from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Sequence

from src.preprocessing.chunking import ChunkConfig


def hash_chunk(chunk_config: ChunkConfig | str, chunk_text: str) -> str:
    return hash_components(chunk_config, chunk_text)


# chunk_hashes are ordered
def hash_section(section_heading: str, chunk_hashes: Sequence[str]) -> str:
    return hash_components(section_heading, list(chunk_hashes))


def hash_document(section_hashes: Sequence[str]) -> str:
    return hash_components(list(section_hashes))


def hash_components(*components: Any) -> str:
    hasher = hashlib.sha256()

    for comp in components:
        serialized = json.dumps(
            get_hashable(comp), sort_keys=True, separators=(",", ":")
        )
        # turns to bytes
        payload = serialized.encode("utf-8")
        hasher.update(str(len(payload)).encode("ascii"))
        hasher.update(b":")
        hasher.update(payload)
        hasher.update(b"|")

    return hasher.hexdigest()


def get_hashable(value: Any) -> Any:
    if isinstance(value, ChunkConfig):
        return value.to_string()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    # sqlite classes
    if is_dataclass(value) and not isinstance(value, type):
        return get_hashable(asdict(value))
    # deep convert dict
    if isinstance(value, dict):
        return {str(key): get_hashable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [get_hashable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    return str(value)
