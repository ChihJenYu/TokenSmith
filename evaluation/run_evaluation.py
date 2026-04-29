#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import shutil
import sqlite3
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from src.config import RAGConfig
from src.incremental import Driver, StateStore
from src.preprocessing.chunking import DocumentChunker
from src.ranking.ranker import EnsembleRanker
from src.retriever import (
    BM25Retriever,
    FAISSRetriever,
    IndexKeywordRetriever,
    load_artifacts,
)


DEFAULT_CONFIG_PATH = "config/config.yaml"
DEFAULT_WORKLOAD_ORCHESTRATION_PATH = "evaluation/workloads.yaml"
DEFAULT_RESULTS_DIR = "evaluation/results"
DEFAULT_RUNTIME_DIR = "evaluation/runtime"
DEFAULT_INDEX_PREFIX = "evaluation"
DEFAULT_RESULT_PREFIX = "evaluation"
SECTION_HEADING_PATTERN = re.compile(
    r"^## (?P<number>\d+(?:\.\d+)*)\s+.+$", re.MULTILINE
)
SECTION_NUMBER_PATTERN = re.compile(r"(\d+(?:\.\d+)*)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workload-orchestration-path",
        default=DEFAULT_WORKLOAD_ORCHESTRATION_PATH,
        help="path to workload orchestration YAML",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="path to config YAML",
    )
    parser.add_argument(
        "--workload-ids",
        default=None,
        help="workload IDs to run, comma-separated. Defaults to all workloads",
    )
    parser.add_argument(
        "--index-prefix",
        default=DEFAULT_INDEX_PREFIX,
        help="index prefix used for this evaluation",
    )
    parser.add_argument(
        "--result-prefix",
        default=DEFAULT_RESULT_PREFIX,
        help="result filename prefix under evaluation/results",
    )
    parser.add_argument(
        "--runtime-dir",
        default=DEFAULT_RUNTIME_DIR,
        help="directory used to store workload corpora.",
    )
    parser.add_argument(
        "--results-dir",
        default=DEFAULT_RESULTS_DIR,
        help="directory for evaluation result files",
    )
    parser.add_argument(
        "--keep-tables",
        action="store_true",
        help="Pass through to the chunker.",
    )
    parser.add_argument(
        "--multiproc-indexing",
        action="store_true",
        help="Use multiprocessing for embeddings",
    )
    parser.add_argument(
        "--embed-with-headings",
        action="store_true",
        help="Prefix chunk text with section headings during indexing",
    )
    return parser.parse_args()


def load_workload_orchestration(
    workload_orchestration_path: Path,
) -> dict[str, Any]:
    with open(workload_orchestration_path, "r", encoding="utf-8") as f:
        workload_orchestration = yaml.safe_load(f)
    if not workload_orchestration:
        raise ValueError(
            f"Workload orchestration file is empty: {workload_orchestration_path}"
        )
    return workload_orchestration


def load_config(config_path: Path) -> RAGConfig:
    return RAGConfig.from_yaml(config_path)


def build_driver(
    cfg: RAGConfig,
    *,
    incremental_mode: bool,
    index_prefix: str,
    keep_tables: bool,
    multiproc_indexing: bool,
    embed_with_headings: bool,
) -> Driver:
    state_store = StateStore(cfg.state_db_path)
    state_store.connect()
    chunker = DocumentChunker(
        strategy=cfg.get_chunk_strategy(),
        keep_tables=keep_tables,
    )
    return Driver(
        state_store=state_store,
        chunker=chunker,
        chunk_config=cfg.chunk_config,
        embedding_model_path=cfg.embed_model,
        artifacts_dir=cfg.get_artifacts_directory(),
        index_prefix=index_prefix,
        use_multiprocessing=multiproc_indexing,
        use_headings=embed_with_headings,
        incremental_mode=incremental_mode,
        enable_metadata_skip=cfg.enable_metadata_skip,
        enable_chunk_reuse=cfg.enable_chunk_reuse,
    )


def remove_existing_state_and_artifacts(cfg: RAGConfig, index_prefix: str) -> None:
    state_db_path = Path(cfg.state_db_path)
    if state_db_path.exists():
        state_db_path.unlink()

    artifacts_dir = Path(cfg.get_artifacts_directory())
    artifact_paths = [
        artifacts_dir / f"{index_prefix}_page_to_chunk_map.json",
        artifacts_dir / f"{index_prefix}.faiss",
        artifacts_dir / f"{index_prefix}_bm25.pkl",
        artifacts_dir / f"{index_prefix}_chunks.pkl",
        artifacts_dir / f"{index_prefix}_sources.pkl",
        artifacts_dir / f"{index_prefix}_meta.pkl",
    ]
    for artifact_path in artifact_paths:
        if artifact_path.exists():
            artifact_path.unlink()


def latest_new_index_log(existing_logs: set[Path]) -> Path:
    current_logs = set(Path("logs").glob("index_*.json"))
    new_logs = sorted(current_logs - existing_logs)
    if not new_logs:
        raise RuntimeError(
            "Expected a new index log to be created, but none was found."
        )
    return new_logs[-1]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def snapshot_active_chunks(state_db_path: Path) -> list[dict[str, Any]]:
    conn = sqlite3.connect(state_db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT d.path AS source_path, c.chunk_hash, c.text, c.metadata_json
            FROM chunks c
            JOIN sections s ON s.id = c.section_id
            JOIN documents d ON d.id = s.document_id
            WHERE c.is_active = 1 AND s.is_active = 1 AND d.is_active = 1
            ORDER BY d.path, s.order_in_parent, c.order_in_parent
            """
        ).fetchall()
    finally:
        conn.close()

    return [
        {
            "source_path": row["source_path"],
            "chunk_hash": row["chunk_hash"],
            "text": row["text"],
            "metadata": json.loads(row["metadata_json"])
            if row["metadata_json"]
            else {},
        }
        for row in rows
    ]


def compare_snapshots(
    left_snapshot: list[dict[str, Any]],
    right_snapshot: list[dict[str, Any]],
) -> dict[str, Any]:
    left_hashes = [entry["chunk_hash"] for entry in left_snapshot]
    right_hashes = [entry["chunk_hash"] for entry in right_snapshot]
    left_texts = [entry["text"] for entry in left_snapshot]
    right_texts = [entry["text"] for entry in right_snapshot]
    left_metadata = [entry["metadata"] for entry in left_snapshot]
    right_metadata = [entry["metadata"] for entry in right_snapshot]

    result = {
        "overall_equal": left_snapshot == right_snapshot,
        "hashes_equal": left_hashes == right_hashes,
        "texts_equal": left_texts == right_texts,
        "metadata_equal": left_metadata == right_metadata,
        "left_chunk_count": len(left_snapshot),
        "right_chunk_count": len(right_snapshot),
        "first_mismatch": None,
    }

    min_len = min(len(left_snapshot), len(right_snapshot))
    for index in range(min_len):
        if left_snapshot[index] != right_snapshot[index]:
            result["first_mismatch"] = {
                "index": index,
                "left": left_snapshot[index],
                "right": right_snapshot[index],
            }
            return result

    if len(left_snapshot) != len(right_snapshot):
        result["first_mismatch"] = {
            "index": min_len,
            "left": left_snapshot[min_len] if len(left_snapshot) > min_len else None,
            "right": right_snapshot[min_len] if len(right_snapshot) > min_len else None,
        }

    return result


def compute_source_to_target_delta(
    source_snapshot: list[dict[str, Any]],
    target_snapshot: list[dict[str, Any]],
) -> dict[str, Any]:
    source_hashes = [entry["chunk_hash"] for entry in source_snapshot]
    target_hashes = [entry["chunk_hash"] for entry in target_snapshot]
    source_hash_set = set(source_hashes)
    target_hash_set = set(target_hashes)
    return {
        "source_active_chunk_count": len(source_snapshot),
        "target_active_chunk_count": len(target_snapshot),
        "shared_chunk_hashes": len(source_hash_set & target_hash_set),
        "removed_chunk_hashes": len(source_hash_set - target_hash_set),
        "added_chunk_hashes": len(target_hash_set - source_hash_set),
        "ordered_hashes_equal": source_hashes == target_hashes,
    }


def document_asset_path(
    manifest: dict[str, Any], project_root: Path, document_name: str
) -> Path:
    document_spec = manifest["assets"]["documents"][document_name]
    relative_path = (
        document_spec["path"] if isinstance(document_spec, dict) else document_spec
    )
    return project_root / relative_path


def snippet_asset_text(
    manifest: dict[str, Any], project_root: Path, snippet_name: str
) -> str:
    snippet_spec = manifest["assets"]["snippets"][snippet_name]
    relative_path = (
        snippet_spec["path"] if isinstance(snippet_spec, dict) else snippet_spec
    )
    return (project_root / relative_path).read_text(encoding="utf-8")


def snippet_asset_format(manifest: dict[str, Any], snippet_name: str) -> str:
    snippet_spec = manifest["assets"]["snippets"][snippet_name]
    if isinstance(snippet_spec, dict):
        return snippet_spec.get("format", "full_markdown")
    return "full_markdown"


def parse_numbered_section_blocks(
    markdown_text: str,
) -> tuple[str, list[dict[str, Any]]]:
    matches = list(SECTION_HEADING_PATTERN.finditer(markdown_text))
    if not matches:
        return markdown_text, []

    preamble = markdown_text[: matches[0].start()]
    blocks: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(markdown_text)
        )
        raw_text = markdown_text[start:end]
        heading_line = raw_text.splitlines()[0]
        body = raw_text[len(heading_line) :].lstrip("\n")
        section_number = match.group("number")
        blocks.append(
            {
                "number": section_number,
                "level": section_number.count(".") + 1,
                "heading_line": heading_line,
                "body": body,
                "raw_text": raw_text,
            }
        )
    return preamble, blocks


def compose_markdown(preamble: str, blocks: list[dict[str, Any]]) -> str:
    return preamble + "".join(block["raw_text"] for block in blocks)


def render_section_block(heading_line: str, body: str) -> str:
    return f"{heading_line}\n\n{body.strip()}\n\n"


def find_section_index(blocks: list[dict[str, Any]], section_number: str) -> int:
    for index, block in enumerate(blocks):
        if block["number"] == section_number:
            return index
    raise ValueError(f"Section {section_number} was not found in the document")


def find_section_subtree_end(blocks: list[dict[str, Any]], start_index: int) -> int:
    prefix = f"{blocks[start_index]['number']}."
    end_index = start_index + 1
    while end_index < len(blocks) and blocks[end_index]["number"].startswith(prefix):
        end_index += 1
    return end_index


def apply_corpus_edit(
    manifest: dict[str, Any],
    project_root: Path,
    document_text: str,
    edit_spec: dict[str, Any],
) -> str:
    edit_type = edit_spec["type"]

    if edit_type == "append_snippet":
        snippet_text = snippet_asset_text(manifest, project_root, edit_spec["snippet"])
        return document_text.rstrip() + "\n\n" + snippet_text.strip() + "\n"

    preamble, blocks = parse_numbered_section_blocks(document_text)
    if not blocks:
        raise ValueError("Expected at least one numbered section block in the document")

    if edit_type == "replace_section_body":
        if snippet_asset_format(manifest, edit_spec["snippet"]) != "body_only":
            raise ValueError("replace_section_body requires a body_only snippet")
        section_index = find_section_index(blocks, edit_spec["section_number"])
        snippet_text = snippet_asset_text(manifest, project_root, edit_spec["snippet"])
        updated_block = dict(blocks[section_index])
        updated_block["body"] = snippet_text.strip()
        updated_block["raw_text"] = render_section_block(
            updated_block["heading_line"],
            snippet_text,
        )
        blocks[section_index] = updated_block
        return compose_markdown(preamble, blocks)

    if edit_type == "remove_section":
        section_index = find_section_index(blocks, edit_spec["section_number"])
        subtree_end = find_section_subtree_end(blocks, section_index)
        del blocks[section_index:subtree_end]
        return compose_markdown(preamble, blocks)

    if edit_type == "swap_sections":
        first_index = find_section_index(blocks, edit_spec["first_section_number"])
        second_index = find_section_index(blocks, edit_spec["second_section_number"])
        if first_index > second_index:
            first_index, second_index = second_index, first_index

        first_end = find_section_subtree_end(blocks, first_index)
        second_end = find_section_subtree_end(blocks, second_index)
        if first_end > second_index:
            raise ValueError("Cannot swap overlapping section ranges")

        reordered_blocks = (
            blocks[:first_index]
            + blocks[second_index:second_end]
            + blocks[first_end:second_index]
            + blocks[first_index:first_end]
            + blocks[second_end:]
        )
        return compose_markdown(preamble, reordered_blocks)

    raise ValueError(f"Unsupported edit type: {edit_type}")


def build_corpus_definition(
    manifest: dict[str, Any],
    project_root: Path,
    corpus_name: str,
    cache: dict[str, dict[str, str]],
) -> dict[str, str]:
    if corpus_name in cache:
        return dict(cache[corpus_name])

    corpus_spec = manifest["corpora"][corpus_name]
    rendered_documents: dict[str, str] = {}

    base_corpus = corpus_spec.get("base")
    if base_corpus:
        rendered_documents.update(
            build_corpus_definition(manifest, project_root, base_corpus, cache)
        )

    for document_name in corpus_spec.get("documents", []):
        document_path = document_asset_path(manifest, project_root, document_name)
        rendered_documents[document_path.name] = document_path.read_text(
            encoding="utf-8"
        )

    for edit_spec in corpus_spec.get("edits", []):
        document_name = edit_spec["document"]
        document_filename = document_asset_path(
            manifest, project_root, document_name
        ).name
        if document_filename not in rendered_documents:
            raise ValueError(
                f"Document {document_name} is not present in corpus {corpus_name}"
            )
        rendered_documents[document_filename] = apply_corpus_edit(
            manifest,
            project_root,
            rendered_documents[document_filename],
            edit_spec,
        )

    cache[corpus_name] = dict(rendered_documents)
    return rendered_documents


def materialize_corpus(
    corpus_documents: dict[str, str], destination_dir: Path
) -> list[Path]:
    if destination_dir.exists():
        shutil.rmtree(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    written_paths: list[Path] = []
    for filename, content in sorted(corpus_documents.items()):
        path = destination_dir / filename
        path.write_text(content, encoding="utf-8")
        written_paths.append(path)
    return written_paths


def select_workloads(
    manifest: dict[str, Any], requested_ids: str | None
) -> list[dict[str, Any]]:
    workloads = manifest.get("workloads", [])
    if not requested_ids:
        return workloads

    requested = {item.strip() for item in requested_ids.split(",") if item.strip()}
    selected = [workload for workload in workloads if workload["id"] in requested]
    missing = sorted(requested - {workload["id"] for workload in selected})
    if missing:
        raise ValueError(f"Unknown workload IDs: {', '.join(missing)}")
    return selected


def select_ablations(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    ablations = manifest.get("ablations", [])
    if not ablations:
        return [
            {
                "id": "metadata_on__reuse_on",
                "enable_metadata_skip": True,
                "enable_chunk_reuse": True,
            }
        ]
    return ablations


def make_ablation_cfg(base_cfg: RAGConfig, ablation_spec: dict[str, Any]) -> RAGConfig:
    cfg = deepcopy(base_cfg)
    cfg.enable_metadata_skip = bool(ablation_spec["enable_metadata_skip"])
    cfg.enable_chunk_reuse = bool(ablation_spec["enable_chunk_reuse"])
    return cfg


def expand_query_sets(
    manifest: dict[str, Any], query_set_names: list[str]
) -> list[dict[str, Any]]:
    query_sets = manifest.get("query_sets", {})
    expanded_queries: list[dict[str, Any]] = []
    seen_query_ids: set[str] = set()

    for query_set_name in query_set_names:
        if query_set_name not in query_sets:
            raise ValueError(f"Unknown query set: {query_set_name}")
        for query_spec in query_sets[query_set_name]:
            query_id = query_spec["id"]
            if query_id in seen_query_ids:
                raise ValueError(f"Duplicate query id in workload: {query_id}")
            seen_query_ids.add(query_id)
            expanded_queries.append(deepcopy(query_spec))

    return expanded_queries


def extract_section_prefix(section_text: str) -> str | None:
    if not section_text:
        return None
    match = SECTION_NUMBER_PATTERN.search(section_text)
    return match.group(1) if match else None


def is_relevant_section(
    retrieved_section_prefix: str | None, relevant_prefixes: list[str]
) -> bool:
    if not retrieved_section_prefix:
        return False
    return any(
        retrieved_section_prefix == relevant_prefix
        or retrieved_section_prefix.startswith(f"{relevant_prefix}.")
        for relevant_prefix in relevant_prefixes
    )


def calculate_recall_at_k(
    retrieved_section_prefixes: list[str | None], relevant_prefixes: list[str]
) -> float:
    if not relevant_prefixes:
        return 0.0
    hits = sum(
        1
        for relevant_prefix in relevant_prefixes
        if any(
            retrieved_prefix == relevant_prefix
            or (
                retrieved_prefix is not None
                and retrieved_prefix.startswith(f"{relevant_prefix}.")
            )
            for retrieved_prefix in retrieved_section_prefixes
        )
    )
    return hits / len(relevant_prefixes)


def calculate_reciprocal_rank(
    retrieved_section_prefixes: list[str | None], relevant_prefixes: list[str]
) -> float:
    for rank, retrieved_prefix in enumerate(retrieved_section_prefixes, start=1):
        if is_relevant_section(retrieved_prefix, relevant_prefixes):
            return 1.0 / rank
    return 0.0


def run_retrieval_queries(
    cfg: RAGConfig,
    *,
    index_prefix: str,
    queries: list[dict[str, Any]],
    top_k: int,
) -> dict[str, Any]:
    artifacts_dir = cfg.get_artifacts_directory()
    faiss_index, bm25_index, chunks, sources, metadata = load_artifacts(
        artifacts_dir=artifacts_dir,
        index_prefix=index_prefix,
    )

    retrievers = [
        FAISSRetriever(faiss_index, cfg.embed_model),
        BM25Retriever(bm25_index),
    ]
    if cfg.ranker_weights.get("index_keywords", 0) > 0:
        retrievers.append(
            IndexKeywordRetriever(cfg.extracted_index_path, cfg.page_to_chunk_map_path)
        )

    ranker = EnsembleRanker(
        ensemble_method=cfg.ensemble_method,
        weights=cfg.ranker_weights,
        rrf_k=int(cfg.rrf_k),
    )

    query_results: list[dict[str, Any]] = []
    for query_spec in queries:
        pool_n = max(cfg.num_candidates, top_k + 10)
        raw_scores: dict[str, dict[int, float]] = {}
        for retriever in retrievers:
            raw_scores[retriever.name] = retriever.get_scores(
                query_spec["text"],
                pool_n,
                chunks,
            )

        ordered_ids, ordered_scores = ranker.rank(raw_scores=raw_scores)
        top_chunk_ids = ordered_ids[:top_k]
        top_scores = ordered_scores[: len(top_chunk_ids)]
        retrieved_chunks: list[dict[str, Any]] = []
        retrieved_section_prefixes: list[str | None] = []

        for chunk_id, score in zip(top_chunk_ids, top_scores):
            chunk_meta = metadata[chunk_id] if chunk_id < len(metadata) else {}
            section_label = chunk_meta.get("section", "")
            section_prefix = extract_section_prefix(
                section_label
            ) or extract_section_prefix(chunk_meta.get("section_path", ""))
            retrieved_section_prefixes.append(section_prefix)
            retrieved_chunks.append(
                {
                    "chunk_id": int(chunk_id),
                    "score": float(score),
                    "section": section_label,
                    "section_prefix": section_prefix,
                    "source": sources[chunk_id],
                }
            )

        relevant_prefixes = query_spec.get("relevant_section_prefixes", [])
        query_results.append(
            {
                "id": query_spec["id"],
                "text": query_spec["text"],
                "relevant_section_prefixes": relevant_prefixes,
                "top_chunk_ids": [int(chunk_id) for chunk_id in top_chunk_ids],
                "top_section_prefixes": retrieved_section_prefixes,
                "recall_at_k": calculate_recall_at_k(
                    retrieved_section_prefixes,
                    relevant_prefixes,
                ),
                "reciprocal_rank": calculate_reciprocal_rank(
                    retrieved_section_prefixes,
                    relevant_prefixes,
                ),
                "retrieved_chunks": retrieved_chunks,
            }
        )

    avg_recall = (
        sum(result["recall_at_k"] for result in query_results) / len(query_results)
        if query_results
        else 0.0
    )
    avg_mrr = (
        sum(result["reciprocal_rank"] for result in query_results) / len(query_results)
        if query_results
        else 0.0
    )

    return {
        "top_k": top_k,
        "queries": query_results,
        "summary": {
            "query_count": len(query_results),
            "avg_recall_at_k": avg_recall,
            "avg_mrr": avg_mrr,
        },
    }


def compare_retrieval_results(
    incremental_results: dict[str, Any],
    clean_results: dict[str, Any],
) -> dict[str, Any]:
    incremental_queries = {
        query_result["id"]: query_result
        for query_result in incremental_results.get("queries", [])
    }
    clean_queries = {
        query_result["id"]: query_result
        for query_result in clean_results.get("queries", [])
    }

    shared_query_ids = sorted(set(incremental_queries) & set(clean_queries))
    query_comparisons: list[dict[str, Any]] = []
    exact_match_count = 0
    overlap_total = 0.0

    for query_id in shared_query_ids:
        incremental_query = incremental_queries[query_id]
        clean_query = clean_queries[query_id]
        incremental_top_ids = incremental_query["top_chunk_ids"]
        clean_top_ids = clean_query["top_chunk_ids"]
        denominator = max(len(clean_top_ids), len(incremental_top_ids), 1)
        overlap = len(set(incremental_top_ids) & set(clean_top_ids)) / denominator
        exact_match = incremental_top_ids == clean_top_ids
        if exact_match:
            exact_match_count += 1
        overlap_total += overlap

        query_comparisons.append(
            {
                "id": query_id,
                "incremental_top_chunk_ids": incremental_top_ids,
                "clean_top_chunk_ids": clean_top_ids,
                "top_k_overlap": overlap,
                "exact_top_k_match": exact_match,
                "incremental_recall_at_k": incremental_query["recall_at_k"],
                "clean_recall_at_k": clean_query["recall_at_k"],
                "incremental_reciprocal_rank": incremental_query["reciprocal_rank"],
                "clean_reciprocal_rank": clean_query["reciprocal_rank"],
            }
        )

    query_count = len(shared_query_ids)
    incremental_summary = incremental_results.get("summary", {})
    clean_summary = clean_results.get("summary", {})
    return {
        "query_count": query_count,
        "exact_top_k_match_rate": exact_match_count / query_count
        if query_count
        else 0.0,
        "avg_top_k_overlap": overlap_total / query_count if query_count else 0.0,
        "incremental_avg_recall_at_k": incremental_summary.get("avg_recall_at_k", 0.0),
        "clean_avg_recall_at_k": clean_summary.get("avg_recall_at_k", 0.0),
        "incremental_avg_mrr": incremental_summary.get("avg_mrr", 0.0),
        "clean_avg_mrr": clean_summary.get("avg_mrr", 0.0),
        "per_query": query_comparisons,
    }


def run_index_once(
    cfg: RAGConfig,
    *,
    runtime_corpus_dir: Path,
    incremental_mode: bool,
    index_prefix: str,
    keep_tables: bool,
    multiproc_indexing: bool,
    embed_with_headings: bool,
    retrieval_queries: list[dict[str, Any]] | None = None,
    retrieval_top_k: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]:
    markdown_paths = sorted(str(path) for path in runtime_corpus_dir.glob("*.md"))
    if not markdown_paths:
        raise FileNotFoundError(
            f"No markdown files found under runtime corpus directory: {runtime_corpus_dir}"
        )

    existing_logs = set(Path("logs").glob("index_*.json"))
    started_at = time.perf_counter()
    driver = build_driver(
        cfg,
        incremental_mode=incremental_mode,
        index_prefix=index_prefix,
        keep_tables=keep_tables,
        multiproc_indexing=multiproc_indexing,
        embed_with_headings=embed_with_headings,
    )
    try:
        driver.run(markdown_paths)
    finally:
        driver.state_store.close()

    elapsed_seconds = time.perf_counter() - started_at
    log_path = latest_new_index_log(existing_logs)
    log_data = read_json(log_path)
    log_data["measured_wall_clock_seconds"] = elapsed_seconds
    snapshot = snapshot_active_chunks(Path(cfg.state_db_path))

    retrieval_results = None
    if retrieval_queries:
        retrieval_results = run_retrieval_queries(
            cfg,
            index_prefix=index_prefix,
            queries=retrieval_queries,
            top_k=retrieval_top_k or cfg.top_k,
        )

    return log_data, snapshot, retrieval_results


def build_ablation_result(
    *,
    ablation_spec: dict[str, Any],
    target_incremental_log: dict[str, Any],
    target_incremental_snapshot: list[dict[str, Any]],
    target_clean_snapshot: list[dict[str, Any]],
    incremental_retrieval: dict[str, Any] | None,
    clean_retrieval: dict[str, Any] | None,
) -> dict[str, Any]:
    target_snapshot_comparison = compare_snapshots(
        target_incremental_snapshot,
        target_clean_snapshot,
    )
    retrieval_comparison = compare_retrieval_results(
        incremental_retrieval or {"queries": [], "summary": {}},
        clean_retrieval or {"queries": [], "summary": {}},
    )
    incremental_metrics = target_incremental_log.get("metrics", {})

    return {
        "settings": {
            "enable_metadata_skip": bool(ablation_spec["enable_metadata_skip"]),
            "enable_chunk_reuse": bool(ablation_spec["enable_chunk_reuse"]),
        },
        "run": deepcopy(target_incremental_log),
        "retrieval": incremental_retrieval,
        "comparisons": {
            "target_incremental_vs_clean": target_snapshot_comparison,
            "retrieval_incremental_vs_clean": retrieval_comparison,
        },
        "summary": {
            "target_incremental_time_seconds": incremental_metrics.get(
                "total_update_time_seconds"
            ),
            "changed_chunks": incremental_metrics.get("changed_chunks"),
            "reused_chunks": incremental_metrics.get("reused_chunks"),
            "re_embedded_chunks": incremental_metrics.get("re_embedded_chunks"),
            "chunk_reuse_rate": incremental_metrics.get("chunk_reuse_rate"),
            "active_chunk_hashes_equal": target_snapshot_comparison["hashes_equal"],
            "active_chunk_texts_equal": target_snapshot_comparison["texts_equal"],
            "active_chunk_metadata_equal": target_snapshot_comparison["metadata_equal"],
            "target_overall_equal": target_snapshot_comparison["overall_equal"],
            "retrieval_exact_top_k_match_rate": retrieval_comparison[
                "exact_top_k_match_rate"
            ],
            "retrieval_avg_top_k_overlap": retrieval_comparison["avg_top_k_overlap"],
            "incremental_avg_recall_at_k": retrieval_comparison[
                "incremental_avg_recall_at_k"
            ],
            "clean_avg_recall_at_k": retrieval_comparison["clean_avg_recall_at_k"],
            "incremental_avg_mrr": retrieval_comparison["incremental_avg_mrr"],
            "clean_avg_mrr": retrieval_comparison["clean_avg_mrr"],
        },
    }


def run_workload(
    cfg: RAGConfig,
    *,
    manifest: dict[str, Any],
    workload_spec: dict[str, Any],
    project_root: Path,
    runtime_root: Path,
    index_prefix: str,
    keep_tables: bool,
    multiproc_indexing: bool,
    embed_with_headings: bool,
    top_k: int,
    ablations: list[dict[str, Any]],
) -> dict[str, Any]:
    corpus_cache: dict[str, dict[str, str]] = {}
    source_documents = build_corpus_definition(
        manifest,
        project_root,
        workload_spec["source_corpus"],
        corpus_cache,
    )
    target_documents = build_corpus_definition(
        manifest,
        project_root,
        workload_spec["target_corpus"],
        corpus_cache,
    )
    retrieval_queries = expand_query_sets(manifest, workload_spec.get("query_sets", []))

    workload_runtime_dir = runtime_root / workload_spec["id"]
    working_corpus_dir = workload_runtime_dir / "corpus"
    workload_runtime_dir.mkdir(parents=True, exist_ok=True)

    remove_existing_state_and_artifacts(cfg, index_prefix)
    materialize_corpus(source_documents, working_corpus_dir)
    source_build_log, source_snapshot, _ = run_index_once(
        cfg,
        runtime_corpus_dir=working_corpus_dir,
        incremental_mode=False,
        index_prefix=index_prefix,
        keep_tables=keep_tables,
        multiproc_indexing=multiproc_indexing,
        embed_with_headings=embed_with_headings,
    )

    remove_existing_state_and_artifacts(cfg, index_prefix)
    materialize_corpus(target_documents, working_corpus_dir)
    target_clean_log, target_clean_snapshot, clean_retrieval = run_index_once(
        cfg,
        runtime_corpus_dir=working_corpus_dir,
        incremental_mode=False,
        index_prefix=index_prefix,
        keep_tables=keep_tables,
        multiproc_indexing=multiproc_indexing,
        embed_with_headings=embed_with_headings,
        retrieval_queries=retrieval_queries,
        retrieval_top_k=top_k,
    )

    source_to_target_delta = compute_source_to_target_delta(
        source_snapshot,
        target_clean_snapshot,
    )
    clean_metrics = target_clean_log.get("metrics", {})
    clean_time = clean_metrics.get("total_update_time_seconds")
    ablation_results: dict[str, Any] = {}

    for ablation_spec in ablations:
        ablation_id = ablation_spec["id"]
        print(
            f"  Running ablation {ablation_id} "
            f"(metadata_skip={ablation_spec['enable_metadata_skip']}, "
            f"chunk_reuse={ablation_spec['enable_chunk_reuse']})..."
        )
        ablation_cfg = make_ablation_cfg(cfg, ablation_spec)

        remove_existing_state_and_artifacts(ablation_cfg, index_prefix)
        materialize_corpus(source_documents, working_corpus_dir)
        _bootstrap_log, _bootstrap_snapshot, _ = run_index_once(
            ablation_cfg,
            runtime_corpus_dir=working_corpus_dir,
            incremental_mode=False,
            index_prefix=index_prefix,
            keep_tables=keep_tables,
            multiproc_indexing=multiproc_indexing,
            embed_with_headings=embed_with_headings,
        )

        materialize_corpus(target_documents, working_corpus_dir)
        target_incremental_log, target_incremental_snapshot, incremental_retrieval = (
            run_index_once(
                ablation_cfg,
                runtime_corpus_dir=working_corpus_dir,
                incremental_mode=True,
                index_prefix=index_prefix,
                keep_tables=keep_tables,
                multiproc_indexing=multiproc_indexing,
                embed_with_headings=embed_with_headings,
                retrieval_queries=retrieval_queries,
                retrieval_top_k=top_k,
            )
        )

        ablation_result = build_ablation_result(
            ablation_spec=ablation_spec,
            target_incremental_log=target_incremental_log,
            target_incremental_snapshot=target_incremental_snapshot,
            target_clean_snapshot=target_clean_snapshot,
            incremental_retrieval=incremental_retrieval,
            clean_retrieval=clean_retrieval,
        )
        incremental_time = ablation_result["summary"][
            "target_incremental_time_seconds"
        ]
        ablation_result["summary"]["incremental_vs_clean_speedup"] = (
            clean_time / incremental_time
            if incremental_time and clean_time and incremental_time > 0
            else None
        )
        ablation_results[ablation_id] = ablation_result

    return {
        "id": workload_spec["id"],
        "description": workload_spec.get("description", ""),
        "source_corpus": workload_spec["source_corpus"],
        "target_corpus": workload_spec["target_corpus"],
        "source_documents": sorted(source_documents.keys()),
        "target_documents": sorted(target_documents.keys()),
        "query_set_names": workload_spec.get("query_sets", []),
        "source_full_build": deepcopy(source_build_log),
        "clean_rebuild": {
            "run": deepcopy(target_clean_log),
            "retrieval": clean_retrieval,
        },
        "comparisons": {
            "source_to_target": source_to_target_delta,
            "target_clean_rebuild": {
                "active_chunk_count": len(target_clean_snapshot),
            },
        },
        "ablations": ablation_results,
    }


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    workload_orchestration_path = (
        project_root / args.workload_orchestration_path
    ).resolve()
    config_path = (project_root / args.config).resolve()
    runtime_root = (project_root / args.runtime_dir).resolve()
    results_dir = (project_root / args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    runtime_root.mkdir(parents=True, exist_ok=True)

    if not workload_orchestration_path.exists():
        raise FileNotFoundError(
            f"Workload orchestration file not found: {workload_orchestration_path}"
        )
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    manifest = load_workload_orchestration(workload_orchestration_path)
    cfg = load_config(config_path)
    workloads = select_workloads(manifest, args.workload_ids)
    ablations = select_ablations(manifest)
    top_k = manifest.get("defaults", {}).get("top_k", cfg.top_k)

    workload_results: dict[str, Any] = {}
    for workload_spec in workloads:
        print(f"Running workload {workload_spec['id']}...")
        workload_result = run_workload(
            cfg,
            manifest=manifest,
            workload_spec=workload_spec,
            project_root=project_root,
            runtime_root=runtime_root,
            index_prefix=args.index_prefix,
            keep_tables=args.keep_tables,
            multiproc_indexing=args.multiproc_indexing,
            embed_with_headings=args.embed_with_headings,
            top_k=top_k,
            ablations=ablations,
        )
        workload_results[workload_spec["id"]] = workload_result

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    result = {
        "timestamp": datetime.now().isoformat(),
        "workload_orchestration_path": str(workload_orchestration_path),
        "config_path": str(config_path),
        "index_prefix": args.index_prefix,
        "top_k": top_k,
        "workload_count": len(workload_results),
        "ablation_ids": [ablation["id"] for ablation in ablations],
    }
    result.update(workload_results)

    output_path = results_dir / f"{args.result_prefix}_{timestamp_str}.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    summary_rows = []
    for workload_id, workload in workload_results.items():
        for ablation_id, ablation_result in workload["ablations"].items():
            summary_rows.append(
                {
                    "workload_id": workload_id,
                    "ablation_id": ablation_id,
                    "speedup": ablation_result["summary"][
                        "incremental_vs_clean_speedup"
                    ],
                    "reuse_rate": ablation_result["summary"]["chunk_reuse_rate"],
                    "retrieval_exact_match_rate": ablation_result["summary"][
                        "retrieval_exact_top_k_match_rate"
                    ],
                    "retrieval_overlap": ablation_result["summary"][
                        "retrieval_avg_top_k_overlap"
                    ],
                    "target_equal": ablation_result["summary"]["target_overall_equal"],
                }
            )

    print(f"Evaluation result written to {output_path}")
    print(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    main()
