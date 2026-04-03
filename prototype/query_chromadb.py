"""
Query ChromaDB Directly: Simple examples of retrieving documents.

Usage:
    python query_chromadb.py "What is the quadratic formula?"
    python query_chromadb.py "Pythagorean theorem"
    python query_chromadb.py --status
    python query_chromadb.py --list
"""

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def print_header(title: str):
    """Print a formatted header."""
    width = 70
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}\n")


def query_chromadb(query: str, k: int = 5):
    """Query ChromaDB and display results."""
    print_header(f"Querying ChromaDB: '{query}'")

    try:
        from prototype.chroma_retrieval import ChromaDBRetriever

        retriever = ChromaDBRetriever()

        # Get status first
        status = retriever.get_status()

        if status["document_count"] == 0:
            logger.warning("⚠️  Collection is empty!")
            logger.warning("Run 'python ingest.py' to load documents first.\n")
            return

        logger.info(f"📊 Collection Status: {status['document_count']} documents\n")

        # Retrieve documents
        logger.info(f"🔍 Retrieving top {k} documents...\n")
        docs = retriever.retrieve(query, k=k)

        if not docs:
            logger.warning("No documents found for this query.")
            return

        # Display results
        for i, doc in enumerate(docs, 1):
            title = doc.get("title", "Unknown")
            source = doc.get("source", "unknown")
            similarity = doc.get("similarity")
            excerpt = doc.get("excerpt", "")
            chunk_num = doc.get("chunk_num", 0)

            # Result header
            logger.info(f"[Result {i}]")
            logger.info(f"  Title: {title}")
            logger.info(f"  Source: {source}")

            if chunk_num > 0:
                logger.info(f"  Chunk: {chunk_num}")

            if similarity is not None:
                # Color similarity based on score
                score_pct = f"{similarity:.1%}"
                if similarity > 0.8:
                    indicator = "🟢"  # High
                elif similarity > 0.6:
                    indicator = "🟡"  # Medium
                else:
                    indicator = "🔴"  # Low
                logger.info(f"  Similarity: {indicator} {score_pct}")

            logger.info(f"\n  Excerpt:")
            # Indent excerpt
            for line in excerpt.split("\n"):
                logger.info(f"    {line}")

            logger.info("")

    except Exception as e:
        logger.error(f"✗ Query failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def show_collection_status():
    """Show collection status and statistics."""
    print_header("ChromaDB Collection Status")

    try:
        from prototype.chroma_retrieval import ChromaDBRetriever

        retriever = ChromaDBRetriever()
        status = retriever.get_status()

        logger.info(f"Status: {status.get('status', 'unknown')}")
        logger.info(f"Documents: {status.get('document_count', 0)}")

        metadata = status.get("metadata", {})
        if metadata:
            logger.info(f"\nMetadata:")
            for key, value in metadata.items():
                logger.info(f"  {key}: {value}")

        if status.get("document_count", 0) == 0:
            logger.warning("\n⚠️  Collection is empty!")
            logger.info("To load documents:")
            logger.info("  1. python ingest.py")

    except Exception as e:
        logger.error(f"✗ Status check failed: {e}")
        sys.exit(1)


def list_documents():
    """List all documents in collection."""
    print_header("Listing All Documents in Collection")

    try:
        from prototype.chroma_retrieval import ChromaDBRetriever

        retriever = ChromaDBRetriever()
        status = retriever.get_status()

        count = status.get("document_count", 0)
        logger.info(f"Total documents/chunks: {count}\n")

        if count == 0:
            logger.warning("Collection is empty!")
            return

        # Query for all documents (retrieve all)
        # Note: This retrieves all docs per query - not ideal for large DBs
        logger.info("(Getting first 100 results...)\n")

        # Use a generic query to get all chunks
        docs = retriever.kb_client.collection.get(limit=100)

        if not docs or "ids" not in docs:
            logger.info("No documents to display")
            return

        for i, doc_id in enumerate(docs["ids"], 1):
            text = docs.get("documents", [""])[i - 1][:100]
            metadata = docs.get("metadatas", [{}])[i - 1]

            logger.info(f"[{i}] ID: {doc_id}")
            logger.info(f"    Text: {text}...")
            if metadata:
                logger.info(f"    Metadata: {metadata}")
            logger.info("")

    except Exception as e:
        logger.error(f"✗ Failed to list documents: {e}")
        sys.exit(1)


def demo_queries():
    """Run some example queries."""
    print_header("Demo Queries")

    example_queries = [
        "What is the Pythagorean theorem?",
        "Solve x^2 + 3x + 2 = 0",
        "Quadratic formula",
    ]

    for query in example_queries:
        query_chromadb(query, k=2)
        logger.info("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Query ChromaDB vector database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python query_chromadb.py "What is algebra?"
  python query_chromadb.py "Pythagorean theorem" -k 10
  python query_chromadb.py --status
  python query_chromadb.py --list
  python query_chromadb.py --demo
        """,
    )

    parser.add_argument("query", nargs="?", help="Query string")
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)",
    )
    parser.add_argument("--status", action="store_true", help="Show collection status")
    parser.add_argument("--list", action="store_true", help="List all documents")
    parser.add_argument("--demo", action="store_true", help="Run demo queries")

    args = parser.parse_args()

    try:
        if args.status:
            show_collection_status()

        elif args.list:
            list_documents()

        elif args.demo:
            demo_queries()

        elif args.query:
            query_chromadb(args.query, k=args.top_k)

        else:
            # Show help if no args
            parser.print_help()
            logger.info("\nTip: Run 'python ingest.py' first to load documents!")

    except KeyboardInterrupt:
        logger.info("\n\nInterrupted.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
