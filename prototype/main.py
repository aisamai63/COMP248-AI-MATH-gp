"""
Main entry point for testing the LangGraph workflow outside of Streamlit.

This script allows you to:
1. Test the workflow programmatically
2. See detailed execution logs
3. Benchmark execution times
4. Debug the multi-agent system

Usage:
    python main.py "What is the quadratic formula?"

    Or:
    from main import demo_query
    demo_query("What is a prime number?")
"""

import sys
import pathlib
import logging
import json
from typing import Dict, Any

# Setup path
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prototype.workflow import create_graph

# Configure logging (detailed output for debugging)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)


def demo_query(query: str) -> Dict[str, Any]:
    """
    Run a demo query through the workflow.

    Args:
        query: User query

    Returns:
        Final workflow state
    """
    logger.info("=" * 80)
    logger.info(f"Starting workflow for query: {query}")
    logger.info("=" * 80)

    # Create graph
    graph = create_graph()

    # Run workflow
    result = graph.run(query)

    # Display results
    _display_results(query, result)

    return result


def _display_results(query: str, result: Dict[str, Any]) -> None:
    """
    Pretty-print workflow results.

    Args:
        query: Original query
        result: Workflow state (final)
    """
    print("\n" + "=" * 80)
    print("WORKFLOW RESULTS")
    print("=" * 80)

    # Query
    print(f"\n🔍 QUERY:\n{query}\n")

    # Summary
    print(f"📝 ANSWER:\n{result.get('summary', '[No summary]')}\n")

    # Retrieved documents
    docs = result.get("retrieved_docs", [])
    print(f"📚 RETRIEVED DOCUMENTS ({len(docs)}):")
    for i, doc in enumerate(docs, 1):
        print(
            f"  {i}. {doc.get('title', 'Untitled')} (source: {doc.get('source', '?')})"
        )
    print()

    # Reflection metrics
    metrics = result.get("reflection_metrics", {})
    if metrics:
        print("📊 REFLECTION METRICS:")
        print(f"  Factual Correctness: {metrics.get('factual_correctness', 0):.1%}")
        print(f"  Completeness:        {metrics.get('completeness', 0):.1%}")
        print(f"  Relevance:           {metrics.get('relevance', 0):.1%}")
        print(f"  Confidence:  {metrics.get('confidence', 0):.1%}")
        if metrics.get("evaluation_source"):
            print(f"  Source:      {metrics.get('evaluation_source')}")

        should_retry = metrics.get("should_retry", False)
        print(f"  Should Retry: {'Yes' if should_retry else 'No'}")
        print()

    # Execution metadata
    metadata = result.get("metadata", {})
    agent_sequence = metadata.get("agent_sequence", [])
    print(f"⚙️  EXECUTION SEQUENCE:")
    print(f"  Agents: {' → '.join(agent_sequence)}")
    print(f"  Iterations: {result.get('iteration_count', 0)}")

    timestamps = metadata.get("timestamps", {})
    if timestamps:
        print(f"  Execution times:")
        total_time = 0
        for agent, elapsed in timestamps.items():
            print(f"    - {agent}: {elapsed:.2f}s")
            total_time += elapsed
        print(f"    - TOTAL: {total_time:.2f}s")
    print()

    # Feedback notes
    if metrics:
        notes = metrics.get("notes", [])
        if notes:
            print("💡 FEEDBACK NOTES:")
            for note in notes:
                print(f"  {note}")
    print()


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Test the Math Inquiries multi-agent workflow"
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="What is the quadratic formula?",
        help="Query to run through the workflow",
    )

    args = parser.parse_args()

    try:
        demo_query(args.query)
    except Exception as e:
        logger.error(f"Workflow failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
