"""Lightweight evaluation script for retrieval + reflection quality.

Run from the repository root:
    python prototype/evaluation.py

The script uses only the Python standard library plus your existing project code.
"""

from __future__ import annotations

import pathlib
import sys
import logging
from statistics import mean
from typing import Any, Dict, List

# Support direct execution from the prototype folder.
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prototype.workflow import create_runtime_graph


logger = logging.getLogger(__name__)


# Input: list of test queries.
TEST_QUERIES = [
    "What is the quadratic formula?",
    "Explain the Pythagorean theorem with one numeric example.",
    "Solve x^2 + 3x + 2 = 0",
]


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _format_score(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value) * 100:.1f}%"
    return "0.0%"


def _mean_similarity(docs: List[Dict[str, Any]]) -> float:
    """Compute mean similarity for retrieved top-k docs (true retrieval metric)."""
    scores = [
        float(doc.get("similarity"))
        for doc in docs
        if isinstance(doc, dict) and isinstance(doc.get("similarity"), (int, float))
    ]
    return mean(scores) if scores else 0.0


def run_evaluation(queries: List[str]) -> List[Dict[str, str]]:
    graph = create_runtime_graph()
    rows: List[Dict[str, str]] = []

    for query in queries:
        try:
            result = graph.run(query)
            docs = result.get("retrieved_docs", [])
            reflection = result.get("reflection_metrics", {})

            rows.append(
                {
                    "query": _truncate(query, 46),
                    "docs": str(len(docs)),
                    "retrieval_metric": _format_score(_mean_similarity(docs)),
                    "answer_metric": _format_score(reflection.get("confidence", 0.0)),
                }
            )
        except Exception as exc:
            logger.warning("Evaluation query failed: %s", query)
            logger.warning("Reason: %s", exc)
            rows.append(
                {
                    "query": _truncate(query, 46),
                    "docs": "0",
                    "retrieval_metric": "0.0%",
                    "answer_metric": "0.0%",
                }
            )

    return rows


def print_table(rows: List[Dict[str, str]]) -> None:
    headers = {
        "query": "Query",
        "docs": "Retrieved Docs",
        "retrieval_metric": "Mean Top-k Similarity",
        "answer_metric": "Reflection Confidence",
    }
    keys = ["query", "docs", "retrieval_metric", "answer_metric"]

    widths = {}
    for key in keys:
        max_content = max([len(row[key]) for row in rows], default=0)
        widths[key] = max(len(headers[key]), max_content)

    def fmt_row(values: Dict[str, str]) -> str:
        return " | ".join(values[k].ljust(widths[k]) for k in keys)

    print(fmt_row(headers))
    print("-+-".join("-" * widths[k] for k in keys))
    for row in rows:
        print(fmt_row(row))


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    rows = run_evaluation(TEST_QUERIES)
    print("Evaluation Results")
    print_table(rows)


if __name__ == "__main__":
    main()
