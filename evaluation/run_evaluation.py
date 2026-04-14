#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import RAGConfig
from src.incremental import Driver, StateStore
from src.preprocessing.chunking import DocumentChunker


DEFAULT_A_SOURCE = "data/Chapter19--extracted_markdown.md"
DEFAULT_B_SOURCE = "evaluation/data/Chapter19--updated-extracted_markdown.md"
DEFAULT_RUNTIME_DIR = "evaluation/runtime"
DEFAULT_RUNTIME_FILENAME = "Chapter19--workload.md"
DEFAULT_RESULTS_DIR = "evaluation/results"
DEFAULT_CONFIG_PATH = "config/config.yaml"
DEFAULT_INDEX_PREFIX = "evaluation"
DEFAULT_RESULT_PREFIX = "evaluation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the TA primary workload end-to-end."
    )
    parser.add_argument(
        "--a-source",
        default=DEFAULT_A_SOURCE,
        help="Original markdown file used to build index A.",
    )
    parser.add_argument(
        "--b-source",
        default=DEFAULT_B_SOURCE,
        help="Updated markdown file used to build index B.",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to config YAML.",
    )
    parser.add_argument(
        "--index-prefix",
        default=DEFAULT_INDEX_PREFIX,
        help="Index prefix used for all runs in this evaluation.",
    )
    parser.add_argument(
        "--result-prefix",
        default=DEFAULT_RESULT_PREFIX,
        help="Result filename prefix under evaluation/results.",
    )
    parser.add_argument(
        "--runtime-dir",
        default=DEFAULT_RUNTIME_DIR,
        help="Temporary markdown corpus directory used during evaluation.",
    )
    parser.add_argument(
        "--runtime-filename",
        default=DEFAULT_RUNTIME_FILENAME,
        help="Runtime markdown filename. A and B are copied here so the path stays stable.",
    )
    parser.add_argument(
        "--results-dir",
        default=DEFAULT_RESULTS_DIR,
        help="Directory for evaluation result files.",
    )
    parser.add_argument(
        "--keep-tables",
        action="store_true",
        help="Pass through to the chunker.",
    )
    parser.add_argument(
        "--multiproc-indexing",
        action="store_true",
        help="Use multiprocessing for embeddings.",
    )
    parser.add_argument(
        "--embed-with-headings",
        action="store_true",
        help="Prefix chunk text with section headings during indexing.",
    )
    return parser.parse_args()


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


def copy_source_to_runtime(source_path: Path, runtime_path: Path) -> None:
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, runtime_path)


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


def compute_a_to_b_delta(
    a_snapshot: list[dict[str, Any]],
    b_snapshot: list[dict[str, Any]],
) -> dict[str, Any]:
    a_hashes = [entry["chunk_hash"] for entry in a_snapshot]
    b_hashes = [entry["chunk_hash"] for entry in b_snapshot]
    a_hash_set = set(a_hashes)
    b_hash_set = set(b_hashes)
    return {
        "a_active_chunk_count": len(a_snapshot),
        "b_active_chunk_count": len(b_snapshot),
        "shared_chunk_hashes": len(a_hash_set & b_hash_set),
        "removed_chunk_hashes": len(a_hash_set - b_hash_set),
        "added_chunk_hashes": len(b_hash_set - a_hash_set),
        "ordered_hashes_equal": a_hashes == b_hashes,
    }


def run_index_once(
    cfg: RAGConfig,
    *,
    runtime_markdown_path: Path,
    incremental_mode: bool,
    index_prefix: str,
    keep_tables: bool,
    multiproc_indexing: bool,
    embed_with_headings: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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
        driver.run([str(runtime_markdown_path)])
    finally:
        driver.state_store.close()

    elapsed_seconds = time.perf_counter() - started_at
    log_path = latest_new_index_log(existing_logs)
    log_data = read_json(log_path)
    log_data["measured_wall_clock_seconds"] = elapsed_seconds
    snapshot = snapshot_active_chunks(Path(cfg.state_db_path))
    return log_data, snapshot


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    a_source_path = Path(args.a_source)
    b_source_path = Path(args.b_source)
    runtime_dir = Path(args.runtime_dir)
    runtime_path = runtime_dir / args.runtime_filename
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if not a_source_path.exists():
        raise FileNotFoundError(f"A source markdown file not found: {a_source_path}")
    if not b_source_path.exists():
        raise FileNotFoundError(f"B source markdown file not found: {b_source_path}")

    cfg = load_config(config_path)
    remove_existing_state_and_artifacts(cfg, args.index_prefix)

    copy_source_to_runtime(a_source_path, runtime_path)
    build_a_log, a_snapshot = run_index_once(
        cfg,
        runtime_markdown_path=runtime_path,
        incremental_mode=False,
        index_prefix=args.index_prefix,
        keep_tables=args.keep_tables,
        multiproc_indexing=args.multiproc_indexing,
        embed_with_headings=args.embed_with_headings,
    )

    copy_source_to_runtime(b_source_path, runtime_path)
    build_b_incremental_log, b_incremental_snapshot = run_index_once(
        cfg,
        runtime_markdown_path=runtime_path,
        incremental_mode=True,
        index_prefix=args.index_prefix,
        keep_tables=args.keep_tables,
        multiproc_indexing=args.multiproc_indexing,
        embed_with_headings=args.embed_with_headings,
    )

    remove_existing_state_and_artifacts(cfg, args.index_prefix)
    build_b_clean_log, b_clean_snapshot = run_index_once(
        cfg,
        runtime_markdown_path=runtime_path,
        incremental_mode=False,
        index_prefix=args.index_prefix,
        keep_tables=args.keep_tables,
        multiproc_indexing=args.multiproc_indexing,
        embed_with_headings=args.embed_with_headings,
    )

    b_comparison = compare_snapshots(b_incremental_snapshot, b_clean_snapshot)
    a_to_b_delta = compute_a_to_b_delta(a_snapshot, b_incremental_snapshot)

    incremental_metrics = build_b_incremental_log.get("metrics", {})
    clean_metrics = build_b_clean_log.get("metrics", {})
    incremental_time = incremental_metrics.get("total_update_time_seconds")
    clean_time = clean_metrics.get("total_update_time_seconds")
    speedup = None
    if incremental_time and clean_time and incremental_time > 0:
        speedup = clean_time / incremental_time

    result = {
        "timestamp": datetime.now().isoformat(),
        "workload": {
            "a_source_path": str(a_source_path),
            "b_source_path": str(b_source_path),
            "runtime_markdown_path": str(runtime_path),
            "config_path": str(config_path),
            "index_prefix": args.index_prefix,
        },
        "runs": {
            "A_full_build": deepcopy(build_a_log),
            "B_incremental": deepcopy(build_b_incremental_log),
            "B_clean_rebuild": deepcopy(build_b_clean_log),
        },
        "comparisons": {
            "A_to_B_incremental": a_to_b_delta,
            "B_incremental_vs_B_clean": b_comparison,
        },
        "summary": {
            "a_build_time_seconds": build_a_log["metrics"].get(
                "total_update_time_seconds"
            ),
            "b_incremental_time_seconds": incremental_time,
            "b_clean_rebuild_time_seconds": clean_time,
            "incremental_vs_clean_speedup": speedup,
            "changed_chunks": incremental_metrics.get("changed_chunks"),
            "reused_chunks": incremental_metrics.get("reused_chunks"),
            "re_embedded_chunks": incremental_metrics.get("re_embedded_chunks"),
            "chunk_reuse_rate": incremental_metrics.get("chunk_reuse_rate"),
            "active_chunk_hashes_equal": b_comparison["hashes_equal"],
            "active_chunk_texts_equal": b_comparison["texts_equal"],
            "active_chunk_metadata_equal": b_comparison["metadata_equal"],
            "b_overall_equal": b_comparison["overall_equal"],
        },
    }

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = results_dir / f"{args.result_prefix}_{timestamp_str}.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Evaluation result written to {output_path}")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
