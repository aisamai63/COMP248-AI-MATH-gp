"""
Benchmark retrieval performance by running N queries against the ChromaDB retriever.

Usage:
    python prototype/benchmark_retrieval.py --n 50 --k 5
    python prototype/benchmark_retrieval.py --queries-file prototype/data/query_examples.txt --n 20

The script prints per-query timings and summary statistics (min, median, mean, p95, max).
"""

import argparse
import random
import time
import statistics
import sys
from pathlib import Path
from typing import List

# Ensure repo root is on sys.path so `from prototype...` imports work when
# running this script directly (not as a module).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prototype.chroma_retrieval import ChromaDBRetriever


DEFAULT_QUERIES = [
    "What is the quadratic formula?",
    "What is the Pythagorean theorem?",
    "How do I find the hypotenuse of a right triangle?",
    "What is a matrix?",
    "What is the norm of a vector in R3?",
]


def load_queries_from_file(path: Path) -> List[str]:
    if not path.exists():
        return DEFAULT_QUERIES
    lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines()]
    queries = []
    for line in lines:
        if not line:
            continue
        # Remove leading numbering like "1. "
        cleaned = line
        if ". " in line[:4]:
            parts = line.split(". ", 1)
            if parts[0].isdigit():
                cleaned = parts[1]
        queries.append(cleaned)
    return queries or DEFAULT_QUERIES


def percentile(sorted_list: List[float], perc: float) -> float:
    if not sorted_list:
        return 0.0
    k = (len(sorted_list) - 1) * (perc / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_list) - 1)
    if f == c:
        return sorted_list[int(k)]
    d0 = sorted_list[f] * (c - k)
    d1 = sorted_list[c] * (k - f)
    return d0 + d1


def run_benchmark(queries: List[str], n: int, k: int, verbose: bool):
    retriever = ChromaDBRetriever()

    timings = []
    results_counts = []

    for i in range(n):
        q = random.choice(queries)
        start = time.perf_counter()
        docs = retriever.retrieve(q, k=k)
        elapsed = time.perf_counter() - start
        timings.append(elapsed)
        results_counts.append(len(docs) if docs is not None else 0)
        if verbose:
            print(f"[{i+1}/{n}] {q!r} -> {len(docs)} docs, {elapsed:.3f}s")

    sorted_timings = sorted(timings)
    summary = {
        "count": len(timings),
        "min": min(timings) if timings else 0.0,
        "median": statistics.median(timings) if timings else 0.0,
        "mean": statistics.mean(timings) if timings else 0.0,
        "p95": percentile(sorted_timings, 95.0),
        "max": max(timings) if timings else 0.0,
    }

    print("\nBenchmark summary:")
    print(f"  Queries run: {summary['count']}")
    print(f"  Min:    {summary['min']:.3f}s")
    print(f"  Median: {summary['median']:.3f}s")
    print(f"  Mean:   {summary['mean']:.3f}s")
    print(f"  95p:    {summary['p95']:.3f}s")
    print(f"  Max:    {summary['max']:.3f}s")

    print(
        f"\nResults count stats: min={min(results_counts)}, max={max(results_counts)}, mean={statistics.mean(results_counts):.2f}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark ChromaDB retrieval performance"
    )
    parser.add_argument("--n", type=int, default=30, help="Number of queries to run")
    parser.add_argument("--k", type=int, default=5, help="Top-k to retrieve")
    parser.add_argument(
        "--queries-file",
        type=str,
        default="prototype/data/query_examples.txt",
        help="File with example queries (one per line)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print per-query timings"
    )
    args = parser.parse_args()

    query_file = Path(args.queries_file)
    queries = load_queries_from_file(query_file)

    print(f"Running benchmark: n={args.n}, k={args.k}, queries_file={query_file}")
    run_benchmark(queries, args.n, args.k, args.verbose)


if __name__ == "__main__":
    main()
