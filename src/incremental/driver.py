from __future__ import annotations

import json
from pathlib import Path
from typing import List
import numpy as np

from src.index_builder import (
    PAGE_MARKER_PATTERN,
    ChunkRecord as RetrievalChunkRecord,
    build_retrieval_artifacts,
    embed_chunks,
    extract_chunk_pages,
    persist_to_index,
    tokenizes_chunks,
)
from src.incremental.hashing import hash_chunk, hash_document, hash_section
from src.incremental.prepared_document import PreparedDocument
from src.incremental.prepared_section import PreparedSection
from src.state_store import (
    ChunkRecord as StateChunkRecord,
    DocumentRecord,
    SectionRecord,
    StateStore,
)
from src.preprocessing.chunking import ChunkConfig, DocumentChunker
from src.preprocessing.extraction import extract_sections_from_markdown


DEFAULT_EXCLUSION_KEYWORDS = ["questions", "exercises", "summary", "references"]


class Driver:
    def __init__(
        self,
        *,
        state_store: StateStore,
        chunker: DocumentChunker,
        chunk_config: ChunkConfig,
        embedding_model_path: str,
        artifacts_dir,
        index_prefix: str,
        use_multiprocessing: bool = False,
        use_headings: bool = False,
        incremental_mode: bool = True,
        exclusion_keywords: List[str] | None = None,
    ) -> None:
        self.state_store = state_store
        self.chunker = chunker
        self.chunk_config = chunk_config
        self.embedding_model_path = embedding_model_path
        self.artifacts_dir = artifacts_dir
        self.index_prefix = index_prefix
        self.use_multiprocessing = use_multiprocessing
        self.use_headings = use_headings
        self.incremental_mode = incremental_mode
        self.exclusion_keywords = exclusion_keywords or DEFAULT_EXCLUSION_KEYWORDS

    def run(self, markdown_files: List[str]) -> None:
        markdown_paths = [str(Path(path)) for path in markdown_files]
        self.deactivate_documents(keep_list=markdown_paths)

        for markdown_path in markdown_paths:
            last_modified_at, size = self.get_document_file_state(markdown_path)
            existing_document = self.state_store.get_document_by_path(markdown_path)

            if not self.should_skip_document(
                existing_document,
                last_modified_at=last_modified_at,
                size=size,
            ):
                prepared_document = self.prepare_document(
                    markdown_path,
                    last_modified_at,
                    size,
                )
                # embeddings
                embedding_updates = self.retrieve_or_create_embeddings(
                    prepared_document
                )
                for section_idx, chunk_idx, embedding_blob in embedding_updates:
                    chunk_record = prepared_document.sections[
                        section_idx
                    ].chunk_records[chunk_idx]
                    prepared_document.sections[section_idx].chunk_records[chunk_idx] = (
                        StateChunkRecord(
                            section_id=chunk_record.section_id,
                            order_in_parent=chunk_record.order_in_parent,
                            chunk_hash=chunk_record.chunk_hash,
                            text=chunk_record.text,
                            bm25_tokens=chunk_record.bm25_tokens,
                            embeddding=embedding_blob,
                            metadata_json=chunk_record.metadata_json,
                            is_active=chunk_record.is_active,
                        )
                    )

                # BM25
                tokenization_updates = self.retrieve_or_create_tokenizations(
                    prepared_document
                )
                for section_idx, chunk_idx, bm25_tokens in tokenization_updates:
                    chunk_record = prepared_document.sections[section_idx].chunk_records[chunk_idx]
                    prepared_document.sections[section_idx].chunk_records[chunk_idx] = StateChunkRecord(
                        section_id=chunk_record.section_id,
                        order_in_parent=chunk_record.order_in_parent,
                        chunk_hash=chunk_record.chunk_hash,
                        text=chunk_record.text,
                        bm25_tokens=bm25_tokens,
                        embeddding=chunk_record.embeddding,
                        metadata_json=chunk_record.metadata_json,
                        is_active=chunk_record.is_active,
                    )

                self.persist_to_state_store(prepared_document)

        active_chunk_records = self.load_active_chunks()
        artifacts = build_retrieval_artifacts(active_chunk_records)
        persist_to_index(
            artifacts,
            artifacts_dir=self.artifacts_dir,
            index_prefix=self.index_prefix,
        )

    def get_document_file_state(self, markdown_path: str) -> tuple[float, int]:
        path = Path(markdown_path)
        stat_result = path.stat()
        return stat_result.st_mtime, stat_result.st_size

    def should_skip_document(
        self,
        existing_document,
        *,
        last_modified_at: float,
        size: int,
    ) -> bool:
        # not incremental means full rebuild
        if not self.incremental_mode:
            return False

        if existing_document is None:
            return False

        return (
            existing_document["last_modified_at"] == last_modified_at
            and existing_document["size"] == size
            and existing_document["is_active"] == 1
        )

    # documents, sections, chunks
    def persist_to_state_store(self, prepared_document: PreparedDocument) -> None:
        document_id = self.state_store.upsert_document(
            DocumentRecord(
                path=prepared_document.path,
                last_modified_at=prepared_document.last_modified_at,
                size=prepared_document.size,
                doc_hash=prepared_document.doc_hash,
                is_active=True,
            )
        )

        self.deactivate_existing_sections_and_chunks(document_id)

        for prepared_section in prepared_document.sections:
            section_id = self.state_store.upsert_section(
                SectionRecord(
                    document_id=document_id,
                    order_in_parent=prepared_section.order_in_parent,
                    heading=prepared_section.heading,
                    level=prepared_section.level,
                    section_hash=prepared_section.section_hash,
                    is_active=True,
                )
            )
            self.state_store.mark_chunks_inactive(section_id)

            for chunk_record in prepared_section.chunk_records:
                self.state_store.upsert_chunk(
                    StateChunkRecord(
                        section_id=section_id,
                        order_in_parent=chunk_record.order_in_parent,
                        chunk_hash=chunk_record.chunk_hash,
                        text=chunk_record.text,
                        bm25_tokens=chunk_record.bm25_tokens,
                        embeddding=chunk_record.embeddding,
                        metadata_json=chunk_record.metadata_json,
                        is_active=True,
                    )
                )

    def deactivate_existing_sections_and_chunks(self, document_id: int) -> None:
        existing_sections = self.state_store.get_sections_for_document(document_id)
        existing_section_ids = [row["id"] for row in existing_sections]
        self.state_store.mark_sections_inactive(document_id)
        for section_id in existing_section_ids:
            self.state_store.mark_chunks_inactive(section_id)

    def prepare_document(
        self,
        markdown_path: str,
        last_modified_at: float,
        size: int,
    ) -> PreparedDocument:
        sections = extract_sections_from_markdown(
            markdown_path,
            exclusion_keywords=self.exclusion_keywords,
        )

        prepared_sections: List[PreparedSection] = []
        current_page = 1
        heading_stack: List[tuple[int, str]] = []

        for section in sections:
            current_level = section.get("level", 1)
            chapter_num = section.get("chapter", 0)

            while heading_stack and heading_stack[-1][0] >= current_level:
                heading_stack.pop()

            if section["heading"] != "Introduction":
                heading_stack.append((current_level, section["heading"]))

            if section["heading"] == "Introduction":
                continue

            path_list = [heading for _, heading in heading_stack]
            full_section_path = " ".join(path_list)
            full_section_path = f"Chapter {chapter_num} " + full_section_path

            raw_chunks = self.chunker.chunk(section["content"])
            state_chunks: List[StateChunkRecord] = []
            chunk_hashes: List[str] = []

            for chunk_order, raw_chunk in enumerate(raw_chunks):
                page_numbers, current_page = extract_chunk_pages(
                    raw_chunk, current_page
                )
                clean_chunk = PAGE_MARKER_PATTERN.sub("", raw_chunk).strip()

                chunk_prefix = ""
                if self.use_headings:
                    chunk_prefix = f"Description: {full_section_path} Content: "

                chunk_text = chunk_prefix + clean_chunk
                chunk_hash = hash_chunk(self.chunk_config, chunk_text)
                chunk_hashes.append(chunk_hash)
                state_chunks.append(
                    StateChunkRecord(
                        section_id=-1,
                        order_in_parent=chunk_order,
                        chunk_hash=chunk_hash,
                        text=chunk_text,
                        bm25_tokens=None,
                        embeddding=None,
                        metadata_json={
                            "filename": markdown_path,
                            "mode": self.chunk_config.to_string(),
                            "char_len": len(clean_chunk),
                            "word_len": len(clean_chunk.split()),
                            "section": section["heading"],
                            "section_path": full_section_path,
                            "text_preview": clean_chunk[:100],
                            "page_numbers": page_numbers,
                        },
                        is_active=True,
                    )
                )

            section_hash = hash_section(section["heading"], chunk_hashes)
            prepared_sections.append(
                PreparedSection(
                    order_in_parent=len(prepared_sections),
                    heading=section["heading"],
                    level=section.get("level", 1),
                    section_hash=section_hash,
                    chunk_records=state_chunks,
                )
            )

        doc_hash = hash_document(
            [section.section_hash for section in prepared_sections]
        )
        return PreparedDocument(
            path=markdown_path,
            last_modified_at=last_modified_at,
            size=size,
            doc_hash=doc_hash,
            sections=prepared_sections,
        )

    def retrieve_or_create_embeddings(
        self,
        prepared_document: PreparedDocument,
    ) -> List[tuple[int, int, bytes]]:
        embedding_updates: List[tuple[int, int, bytes]] = []
        missing_chunks: List[tuple[int, int, StateChunkRecord]] = []

        for section_idx, prepared_section in enumerate(prepared_document.sections):
            for chunk_idx, chunk_record in enumerate(prepared_section.chunk_records):
                cached_chunk = self.state_store.get_chunk_by_hash(
                    chunk_record.chunk_hash
                )
                if cached_chunk is not None and cached_chunk["embeddding"] is not None:
                    embedding_updates.append(
                        (section_idx, chunk_idx, cached_chunk["embeddding"])
                    )
                else:
                    missing_chunks.append((section_idx, chunk_idx, chunk_record))

        if missing_chunks:
            missing_texts = [chunk_record.text for _, _, chunk_record in missing_chunks]
            computed_embeddings = embed_chunks(
                missing_texts,
                embedding_model_path=self.embedding_model_path,
                use_multiprocessing=self.use_multiprocessing,
            )
            for (section_idx, chunk_idx, _chunk_record), embedding in zip(
                missing_chunks, computed_embeddings
            ):
                embedding_updates.append(
                    (
                        section_idx,
                        chunk_idx,
                        np.asarray(embedding, dtype=np.float32).tobytes(),
                    )
                )

        return embedding_updates

    def retrieve_or_create_tokenizations(
        self,
        prepared_document: PreparedDocument,
    ) -> List[tuple[int, int, List[str]]]:
        tokenization_updates: List[tuple[int, int, List[str]]] = []
        missing_chunks: List[tuple[int, int, StateChunkRecord]] = []

        for section_idx, prepared_section in enumerate(prepared_document.sections):
            for chunk_idx, chunk_record in enumerate(prepared_section.chunk_records):
                cached_chunk = self.state_store.get_chunk_by_hash(
                    chunk_record.chunk_hash
                )
                if cached_chunk is not None and cached_chunk["bm25_tokens"] is not None:
                    tokenization_updates.append(
                        (
                            section_idx,
                            chunk_idx,
                            json.loads(cached_chunk["bm25_tokens"]),
                        )
                    )
                else:
                    missing_chunks.append((section_idx, chunk_idx, chunk_record))

        if missing_chunks:
            missing_texts = [chunk_record.text for _, _, chunk_record in missing_chunks]
            computed_tokens = tokenizes_chunks(missing_texts)

            for (section_idx, chunk_idx, _chunk_record), tokens in zip(
                missing_chunks, computed_tokens
            ):
                tokenization_updates.append((section_idx, chunk_idx, tokens))

        return tokenization_updates

    def deactivate_documents(self, keep_list: List[str]) -> None:
        current_paths = set(keep_list)
        for document_row in self.state_store.get_all_documents(is_active=True):
            if document_row["path"] in current_paths:
                continue

            self.state_store.mark_document_inactive(document_row["path"])
            existing_sections = self.state_store.get_sections_for_document(
                document_row["id"]
            )
            self.state_store.mark_sections_inactive(document_row["id"])
            for section_row in existing_sections:
                self.state_store.mark_chunks_inactive(section_row["id"])

    def load_active_chunks(self) -> List[RetrievalChunkRecord]:
        active_records: List[RetrievalChunkRecord] = []
        for row in self.state_store.get_all_active_chunks():
            metadata_json = row["metadata_json"]
            metadata = json.loads(metadata_json) if metadata_json else {}
            bm25_tokens_json = row["bm25_tokens"]
            embedding_blob = row["embeddding"]
            active_records.append(
                RetrievalChunkRecord(
                    text=row["text"],
                    source=row["source_path"],
                    metadata=metadata,
                    bm25_tokens=json.loads(bm25_tokens_json)
                    if bm25_tokens_json
                    else None,
                    embedding=np.frombuffer(embedding_blob, dtype=np.float32)
                    if embedding_blob
                    else None,
                )
            )
        return active_records
