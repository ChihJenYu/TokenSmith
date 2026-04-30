#!/usr/bin/env python3
"""
Log analysis utility for RAG pipeline runs.
Provides detailed insights into retrieval, ranking, and generation performance.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any
import statistics


def load_session_logs(session_id: str) -> List[Dict[str, Any]]:
    """Load all log entries for a session."""
    log_file = Path("logs") / f"run_{session_id}.jsonl"
    if not log_file.exists():
        print(f"Log file not found: {log_file}")
        return []

    logs = []
    with open(log_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                logs.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse line {line_num}: {e}")
                continue
    return logs


def load_index_logs() -> List[Dict[str, Any]]:
    logs = []
    for log_file in sorted(Path("logs").glob("index_*.json")):
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                log_data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse {log_file.name}: {e}")
            continue

        if log_data.get("event") == "index_run":
            log_data["_log_file"] = str(log_file)
            logs.append(log_data)
    return logs


def analyze_index_runs(index_runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    metric_names = [
        "total_documents_scanned",
        "changed_documents",
        "changed_sections",
        "changed_chunks",
        "reused_chunks",
        "re_embedded_chunks",
        "chunk_reuse_rate",
        "embedding_time_seconds",
        "total_update_time_seconds",
    ]

    metric_values: Dict[str, List[float]] = {metric_name: [] for metric_name in metric_names}
    modes = Counter()

    for run in index_runs:
        modes[run.get("mode", "unknown")] += 1
        metrics = run.get("metrics", {})
        for metric_name in metric_names:
            metric_value = metrics.get(metric_name)
            if metric_value is not None:
                metric_values[metric_name].append(metric_value)

    summaries = {}
    for metric_name, values in metric_values.items():
        if not values:
            continue
        summaries[metric_name] = {
            "latest": values[-1],
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }

    return {
        "run_count": len(index_runs),
        "modes": dict(modes),
        "metrics": summaries,
        "latest_log_file": index_runs[-1].get("_log_file") if index_runs else None,
    }


def analyze_retrieval_performance(queries: List[Dict]) -> Dict[str, Any]:
    """Analyze FAISS retrieval performance."""
    distances = []
    pool_sizes = []
    candidates_returned = []

    for query in queries:
        if "retrieval" in query:
            ret = query["retrieval"]
            pool_sizes.append(ret.get("pool_size_requested", 0))
            candidates_returned.append(ret.get("candidates_returned", 0))

            if ret.get("faiss_stats"):
                stats = ret["faiss_stats"]
                if stats.get("avg_distance"):
                    distances.append(stats["avg_distance"])

    return {
        "avg_pool_size": statistics.mean(pool_sizes) if pool_sizes else 0,
        "avg_candidates_returned": statistics.mean(candidates_returned) if candidates_returned else 0,
        "avg_faiss_distance": statistics.mean(distances) if distances else 0,
        "distance_std": statistics.stdev(distances) if len(distances) > 1 else 0,
        "retrieval_efficiency": statistics.mean([c / p for c, p in zip(candidates_returned, pool_sizes) if
                                                 p > 0]) if candidates_returned and pool_sizes else 0
    }


def analyze_ranker_performance(queries: List[Dict]) -> Dict[str, Any]:
    """Analyze individual ranker performance and consistency."""
    ranker_stats = defaultdict(lambda: {
        "usage_count": 0,
        "scores": [],
        "nonzero_scores": [],
        "rank_positions": defaultdict(int)  # How often each ranker puts items in top positions
    })

    for query in queries:
        if "ranking" not in query:
            continue

        for ranker_name, ranker_data in query["ranking"].items():
            stats = ranker_stats[ranker_name]
            stats["usage_count"] += 1

            scores = list(ranker_data["scores"].values())
            stats["scores"].extend(scores)
            stats["nonzero_scores"].extend([s for s in scores if s > 0])

            # Analyze ranking positions
            ranks = ranker_data.get("ranks", {})
            for idx, rank in ranks.items():
                if rank <= 5:  # Top 5 positions
                    stats["rank_positions"][f"top_{rank}"] += 1

    # Compute final statistics
    final_stats = {}
    for ranker, data in ranker_stats.items():
        scores = data["scores"]
        nonzero = data["nonzero_scores"]

        final_stats[ranker] = {
            "usage_count": data["usage_count"],
            "avg_score": statistics.mean(scores) if scores else 0,
            "score_std": statistics.stdev(scores) if len(scores) > 1 else 0,
            "nonzero_ratio": len(nonzero) / len(scores) if scores else 0,
            "avg_nonzero_score": statistics.mean(nonzero) if nonzero else 0,
            "top_1_frequency": data["rank_positions"]["top_1"],
            "top_3_frequency": sum(data["rank_positions"][f"top_{i}"] for i in [1, 2, 3]),
            "top_5_frequency": sum(data["rank_positions"][f"top_{i}"] for i in range(1, 6))
        }

    return final_stats


def analyze_ensemble_consistency(queries: List[Dict]) -> Dict[str, Any]:
    """Analyze how consistent the ensemble method is."""
    methods_used = Counter()
    final_rankings = []

    for query in queries:
        if "ensemble" in query:
            ens = query["ensemble"]
            methods_used[ens.get("method", "unknown")] += 1

            final_rank = ens.get("final_ranking", [])
            if len(final_rank) >= 3:
                # Check stability of top 3
                final_rankings.append(final_rank[:3])

    return {
        "methods_used": dict(methods_used),
        "total_rankings": len(final_rankings),
        "avg_top_3_diversity": len(set(sum(final_rankings, []))) / max(len(final_rankings) * 3,
                                                                       1) if final_rankings else 0
    }


def analyze_generation_patterns(queries: List[Dict]) -> Dict[str, Any]:
    """Analyze generation performance and patterns."""
    response_lengths = []
    prompt_lengths = []
    generation_times = []

    for query in queries:
        if "generation" not in query:
            continue

        gen = query["generation"]
        response_lengths.append(gen.get("response_char_length", 0))

        if gen.get("prompt_length_estimate"):
            prompt_lengths.append(gen["prompt_length_estimate"])

    return {
        "avg_response_length": statistics.mean(response_lengths) if response_lengths else 0,
        "response_length_std": statistics.stdev(response_lengths) if len(response_lengths) > 1 else 0,
        "avg_prompt_length": statistics.mean(prompt_lengths) if prompt_lengths else 0,
        "response_length_range": [min(response_lengths), max(response_lengths)] if response_lengths else [0, 0],
        "total_responses": len(response_lengths)
    }


def analyze_query_patterns(queries: List[Dict]) -> Dict[str, Any]:
    """Analyze patterns in user queries."""
    query_lengths = []
    query_types = defaultdict(int)

    for query in queries:
        if "query" not in query:
            continue

        q_text = query["query"]
        query_lengths.append(len(q_text))

        # Simple query classification
        q_lower = q_text.lower()
        if any(term in q_lower for term in ["what is", "define", "definition"]):
            query_types["definition"] += 1
        elif any(term in q_lower for term in ["how to", "how do", "steps"]):
            query_types["procedural"] += 1
        elif any(term in q_lower for term in ["why", "explain", "because"]):
            query_types["explanatory"] += 1
        elif "?" in q_text:
            query_types["question"] += 1
        else:
            query_types["other"] += 1

    return {
        "total_queries": len(queries),
        "avg_query_length": statistics.mean(query_lengths) if query_lengths else 0,
        "query_length_std": statistics.stdev(query_lengths) if len(query_lengths) > 1 else 0,
        "query_types": dict(query_types),
        "query_length_range": [min(query_lengths), max(query_lengths)] if query_lengths else [0, 0]
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze RAG pipeline logs")
    parser.add_argument("--session_id", help="Session ID to analyze", default="20250918_004127")
    parser.add_argument(
        "--log_type",
        choices=["chat", "index"],
        default="chat",
        help="Which logs to analyze",
    )
    parser.add_argument("--detailed", action="store_true", help="Show detailed analysis")
    parser.add_argument("--export-json", help="Export analysis to JSON file")

    args = parser.parse_args()

    if args.log_type == "index":
        index_runs = load_index_logs()
        if not index_runs:
            print("No index-run logs found in logs/.")
            return

        analysis = analyze_index_runs(index_runs)
        print("\n=== INDEX RUN ANALYSIS ===\n")
        print(f"Runs analyzed: {analysis['run_count']}")
        print(f"Modes: {analysis['modes']}")
        if analysis["latest_log_file"]:
            print(f"Latest log: {analysis['latest_log_file']}")
        print()

        for metric_name, summary in analysis["metrics"].items():
            print(f"{metric_name}:")
            print(f"  latest: {summary['latest']}")
            print(f"  mean: {summary['mean']}")
            print(f"  median: {summary['median']}")
            print(f"  min: {summary['min']}")
            print(f"  max: {summary['max']}")
            print()

        if args.export_json:
            with open(args.export_json, "w", encoding="utf-8") as f:
                json.dump(analysis, f, indent=2)
        return

    # Load logs
    logs = load_session_logs(args.session_id)
    if not logs:
        return

    print(f"\n=== LOG ANALYSIS: {args.session_id} ===\n")

    # Separate different types of logs
    session_info = [log for log in logs if log.get("event") == "session_start"]
    queries = [log for log in logs if log.get("event") == "query"]
    errors = [log for log in logs if log.get("event") == "error"]

    if session_info:
        print("SESSION CONFIG:")
        config = session_info[0].get("config", {})
        for key, value in config.items():
            print(f"  {key}: {value}")
        print()

    if not queries:
        print("No query logs found in this session.")
        return

    print(f"SUMMARY:")
    print(f"  Total queries: {len(queries)}")
    print(f"  Errors: {len(errors)}")
    print(f"  Success rate: {((len(queries) - len(errors)) / len(queries) * 100):.1f}%")
    print()

    # Analyze different aspects
    retrieval_analysis = analyze_retrieval_performance(queries)
    ranker_analysis = analyze_ranker_performance(queries)
    ensemble_analysis = analyze_ensemble_consistency(queries)
    generation_analysis = analyze_generation_patterns(queries)
    query_analysis = analyze_query_patterns(queries)

    # Print retrieval analysis
    print("RETRIEVAL PERFORMANCE:")
    print(f"  Avg pool size: {retrieval_analysis['avg_pool_size']:.1f}")
    print(f"  Avg candidates returned: {retrieval_analysis['avg_candidates_returned']:.1f}")
    print(f"  Avg FAISS distance: {retrieval_analysis['avg_faiss_distance']:.3f}")
    print(f"  Distance std dev: {retrieval_analysis['distance_std']:.3f}")
    print(f"  Retrieval efficiency: {retrieval_analysis['retrieval_efficiency']:.2f}")
    print()

    # Print ranker analysis
    print("RANKER PERFORMANCE:")
    for ranker, stats in ranker_analysis.items():
        print(f"  {ranker}:")
        print(f"    Usage: {stats['usage_count']} queries")
        print(f"    Avg score: {stats['avg_score']:.3f} (±{stats['score_std']:.3f})")
        print(f"    Non-zero ratio: {stats['nonzero_ratio']:.2f}")
        print(f"    Top-1 frequency: {stats['top_1_frequency']}")
        print(f"    Top-3 frequency: {stats['top_3_frequency']}")
    print()

    # Print ensemble analysis
    print("ENSEMBLE ANALYSIS:")
    for method, count in ensemble_analysis["methods_used"].items():
        print(f"  {method}: {count} queries")
    print(f"  Top-3 diversity: {ensemble_analysis['avg_top_3_diversity']:.2f}")
    print()

    # Print generation analysis
    print("GENERATION PERFORMANCE:")
    print(f"  Avg response length: {generation_analysis['avg_response_length']:.0f} chars")
    print(f"  Response length std: {generation_analysis['response_length_std']:.0f}")
    print(f"  Avg prompt length: {generation_analysis['avg_prompt_length']:.0f} chars")
    print(f"  Response length range: {generation_analysis['response_length_range']}")
    print()

    # Print query analysis
    print("QUERY PATTERNS:")
    print(f"  Avg query length: {query_analysis['avg_query_length']:.0f} chars")
    print(f"  Query types:")
    for q_type, count in query_analysis["query_types"].items():
        print(f"    {q_type}: {count}")
    print()

    # Show detailed analysis if requested
    if args.detailed:
        print("DETAILED QUERY BREAKDOWN:")
        for i, query in enumerate(queries, 1):
            print(f"\nQuery {i}: {query.get('query', 'N/A')[:100]}...")
            if "retrieval" in query:
                ret = query["retrieval"]
                print(f"  Retrieved: {ret.get('candidates_returned', 0)} candidates")

            if "ranking" in query:
                print("  Ranker scores:")
                for ranker, data in query["ranking"].items():
                    scores = list(data["scores"].values())
                    if scores:
                        print(f"    {ranker}: avg={statistics.mean(scores):.3f}, max={max(scores):.3f}")

            if "generation" in query:
                gen = query["generation"]
                print(f"  Response: {gen.get('response_char_length', 0)} chars")

    # Export to JSON if requested
    if args.export_json:
        analysis_data = {
            "session_id": args.session_id,
            "summary": {
                "total_queries": len(queries),
                "total_errors": len(errors),
                "success_rate": (len(queries) - len(errors)) / len(queries) if queries else 0
            },
            "retrieval": retrieval_analysis,
            "rankers": ranker_analysis,
            "ensemble": ensemble_analysis,
            "generation": generation_analysis,
            "queries": query_analysis
        }

        with open(args.export_json, "w") as f:
            json.dump(analysis_data, f, indent=2)
        print(f"Analysis exported to: {args.export_json}")


if __name__ == "__main__":
    main()
