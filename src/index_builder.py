#!/usr/bin/env python3
"""
index_builder.py
PDF -> markdown text -> chunks -> embeddings -> BM25 + FAISS + metadata

Entry point (called by main.py):
    build_index(markdown_file, ...)
"""

import json
import os
import pathlib
import pickle
import re
from dataclasses import dataclass
from typing import Any, Dict, List

import faiss
from rank_bm25 import BM25Okapi

from src.embedder import SentenceTransformer
from src.preprocessing.chunking import ChunkConfig, DocumentChunker
from src.preprocessing.extraction import extract_sections_from_markdown

# ----- runtime parallelism knobs (avoid oversubscription) -----
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Default keywords to exclude sections
DEFAULT_EXCLUSION_KEYWORDS = ["questions", "exercises", "summary", "references"]
PAGE_MARKER_PATTERN = re.compile(r"--- Page (\d+) ---")


@dataclass(frozen=True)
class ChunkRecord:
    text: str
    source: str
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class RetrievalArtifacts:
    faiss_index: faiss.Index
    bm25_index: BM25Okapi
    chunks: List[str]
    sources: List[str]
    metadata: List[Dict[str, Any]]
    page_to_chunk_map: Dict[int, List[int]]


# ------------------------ Main index builder -----------------------------

# process docuemnt into chunks -> build embedding artifacts (tokens & embedding) -> persist to state db
def build_index(
    markdown_file: str,
    *,
    chunker: DocumentChunker,
    chunk_config: ChunkConfig,
    embedding_model_path: str,
    artifacts_dir: os.PathLike,
    index_prefix: str,
    use_multiprocessing: bool = False,
    use_headings: bool = False,
) -> None:
    chunk_records = process_markdown_document(
        markdown_file,
        chunker=chunker,
        chunk_config=chunk_config,
        use_headings=use_headings,
        exclusion_keywords=DEFAULT_EXCLUSION_KEYWORDS,
    )

    artifacts = build_retrieval_artifacts(
        chunk_records,
        embedding_model_path=embedding_model_path,
        use_multiprocessing=use_multiprocessing,
    )

    persist_retrieval_artifacts(
        artifacts,
        artifacts_dir=artifacts_dir,
        index_prefix=index_prefix,
    )

# Parse and chunk one markdown document into reusable chunk records
def process_markdown_document(
    markdown_file: str,
    *,
    chunker: DocumentChunker,
    chunk_config: ChunkConfig,
    use_headings: bool = False,
    exclusion_keywords: List[str] | None = None,
) -> List[ChunkRecord]:
    sections = extract_sections_from_markdown(
        markdown_file,
        exclusion_keywords=exclusion_keywords,
    )

    chunk_records: List[ChunkRecord] = []
    current_page = 1
    heading_stack: List[tuple[int, str]] = []

    for section in sections:
        current_level = section.get("level", 1)
        chapter_num = section.get("chapter", 0)

        while heading_stack and heading_stack[-1][0] >= current_level:
            heading_stack.pop()

        if section["heading"] != "Introduction":
            heading_stack.append((current_level, section["heading"]))

        path_list = [heading for _, heading in heading_stack]
        full_section_path = " ".join(path_list)
        full_section_path = f"Chapter {chapter_num} " + full_section_path

        sub_chunks = chunker.chunk(section["content"])

        for sub_chunk in sub_chunks:
            chunk_pages, current_page = extract_chunk_pages(sub_chunk, current_page)
            clean_chunk = PAGE_MARKER_PATTERN.sub("", sub_chunk).strip()

            if section["heading"] == "Introduction":
                continue

            chunk_prefix = ""
            if use_headings:
                chunk_prefix = f"Description: {full_section_path} Content: "

            metadata = {
                "filename": markdown_file,
                "mode": chunk_config.to_string(),
                "char_len": len(clean_chunk),
                "word_len": len(clean_chunk.split()),
                "section": section["heading"],
                "section_path": full_section_path,
                "text_preview": clean_chunk[:100],
                "page_numbers": chunk_pages,
            }

            chunk_records.append(
                ChunkRecord(
                    text=chunk_prefix + clean_chunk,
                    source=markdown_file,
                    metadata=metadata,
                )
            )

    return chunk_records


# Which pages are involved in this chunk
def extract_chunk_pages(sub_chunk: str, current_page: int) -> tuple[List[int], int]:
    chunk_pages = set()
    fragments = PAGE_MARKER_PATTERN.split(sub_chunk)

    if fragments[0].strip():
        chunk_pages.add(current_page)

    for i in range(1, len(fragments), 2):
        try:
            new_page = int(fragments[i]) + 1
            if fragments[i + 1].strip():
                chunk_pages.add(new_page)
            current_page = new_page
        except (IndexError, ValueError):
            continue

    return sorted(chunk_pages), current_page

def embed_chunks(
    chunks: List[str],
    *,
    embedding_model_path: str,
    use_multiprocessing: bool = False,
):
    print(f"Embedding {len(chunks):,} chunks with {pathlib.Path(embedding_model_path).stem} ...")
    embedder = SentenceTransformer(embedding_model_path)

    if use_multiprocessing:
        print("Starting multi-process pool for embeddings...")
        pool = embedder.start_multi_process_pool(num_workers=4)
        try:
            return embedder.encode_multi_process(
                chunks,
                pool,
                batch_size=32,
            )
        finally:
            embedder.stop_multi_process_pool(pool)

    return embedder.encode(
        chunks,
        batch_size=8,
        show_progress_bar=True,
        convert_to_numpy=True,
    )


def tokenizes_chunks(chunks: List[str]) -> List[List[str]]:
    return [preprocess_for_bm25(chunk) for chunk in chunks]

def build_faiss_index(embeddings) -> faiss.Index:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def build_bm25_index(tokenized_chunks: List[List[str]]) -> BM25Okapi:
    return BM25Okapi(tokenized_chunks)

def build_retrieval_artifacts(
    chunk_records: List[ChunkRecord],
    *,
    embedding_model_path: str,
    use_multiprocessing: bool = False,
) -> RetrievalArtifacts:
    chunks: List[str] = []
    sources: List[str] = []
    metadata: List[Dict[str, Any]] = []
    page_to_chunk_map: Dict[int, List[int]] = {}

    for chunk_id, chunk_record in enumerate(chunk_records):
        chunks.append(chunk_record.text)
        sources.append(chunk_record.source)

        meta = dict(chunk_record.metadata)
        meta["chunk_id"] = chunk_id
        metadata.append(meta)

        for page_number in meta.get("page_numbers", []):
            page_to_chunk_map.setdefault(page_number, []).append(chunk_id)

    embeddings = embed_chunks(
        chunks,
        embedding_model_path=embedding_model_path,
        use_multiprocessing=use_multiprocessing,
    )
    tokenized_chunks = tokenizes_chunks(chunks)

    print(f"Building FAISS index for {len(chunks):,} chunks...")
    faiss_index = build_faiss_index(embeddings)
    print(f"Building BM25 index for {len(chunks):,} chunks...")
    bm25_index = build_bm25_index(tokenized_chunks)

    return RetrievalArtifacts(
        faiss_index=faiss_index,
        bm25_index=bm25_index,
        chunks=chunks,
        sources=sources,
        metadata=metadata,
        page_to_chunk_map=page_to_chunk_map,
    )

def persist_retrieval_artifacts(
    artifacts: RetrievalArtifacts,
    *,
    artifacts_dir: os.PathLike,
    index_prefix: str,
) -> None:
    artifacts_dir = pathlib.Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    output_file = artifacts_dir / f"{index_prefix}_page_to_chunk_map.json"
    with open(output_file, "w") as f:
        json.dump(artifacts.page_to_chunk_map, f, indent=2)
    print(f"Saved page to chunk ID map: {output_file}")

    faiss.write_index(artifacts.faiss_index, str(artifacts_dir / f"{index_prefix}.faiss"))
    print(f"FAISS Index built successfully: {index_prefix}.faiss")

    with open(artifacts_dir / f"{index_prefix}_bm25.pkl", "wb") as f:
        pickle.dump(artifacts.bm25_index, f)
    print(f"BM25 Index built successfully: {index_prefix}_bm25.pkl")

    with open(artifacts_dir / f"{index_prefix}_chunks.pkl", "wb") as f:
        pickle.dump(artifacts.chunks, f)
    with open(artifacts_dir / f"{index_prefix}_sources.pkl", "wb") as f:
        pickle.dump(artifacts.sources, f)
    with open(artifacts_dir / f"{index_prefix}_meta.pkl", "wb") as f:
        pickle.dump(artifacts.metadata, f)
    print(f"Saved all index artifacts with prefix: {index_prefix}")

# ------------------------ Helper functions ------------------------------

def preprocess_for_bm25(text: str) -> list[str]:
    """
    Simplifies text to keep only letters, numbers, underscores, hyphens,
    apostrophes, plus, and hash — suitable for BM25 tokenization.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9_'#+-]", " ", text)
    return text.split()
