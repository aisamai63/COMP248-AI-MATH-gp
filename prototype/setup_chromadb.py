"""
ChromaDB RAG Setup: Complete guide to initialize and use the persistent vector database.

This script demonstrates:
1. Initializing ChromaDB with persistent storage
2. Ingesting documents with embeddings
3. Querying the database
4. Checking collection status

Quick Start:
    # 1. Set up ChromaDB
    python setup_chromadb.py

    # 2. Ingest documents
    python ingest.py

    # 3. Test retrieval
    python query_chromadb.py "What is x^2?"

    # 4. Run the full system
    python main.py "your query"
"""

import os
import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_chromadb():
    """Initialize ChromaDB persistent database."""
    logger.info("=" * 70)
    logger.info("🗄️  CHROMADB INITIALIZATION")
    logger.info("=" * 70)

    try:
        from prototype.chroma_setup import (
            get_chroma_client,
            get_chroma_collection,
            check_collection_status,
        )

        logger.info("\n1️⃣  Initializing ChromaDB client...")
        client = get_chroma_client()
        logger.info("   ✓ Client initialized with persistent storage (.chroma_db/)")

        logger.info("\n2️⃣  Creating/getting collection...")
        collection = get_chroma_collection(client)
        logger.info(f"   ✓ Collection 'math_docs' ready")

        logger.info("\n3️⃣  Checking collection status...")
        status = check_collection_status(collection)
        logger.info(f"   Documents: {status['document_count']}")
        logger.info(f"   Status: {status['status']}")

        if status["document_count"] == 0:
            logger.warning("\n   ⚠️  Collection is empty!")
            logger.warning("   Run 'python ingest.py' to load documents")

        return client, collection

    except Exception as e:
        logger.error(f"✗ Failed to setup ChromaDB: {e}")
        raise


def ingest_documents():
    """Ingest documents into ChromaDB."""
    logger.info("\n" + "=" * 70)
    logger.info("📚 DOCUMENT INGESTION")
    logger.info("=" * 70)

    try:
        from prototype.ingest import main as ingest_main

        ingest_main()

    except Exception as e:
        logger.error(f"✗ Ingestion failed: {e}")
        raise


def test_retrieval(query: str):
    """Test ChromaDB retrieval."""
    logger.info("\n" + "=" * 70)
    logger.info("🔍 TESTING RETRIEVAL")
    logger.info("=" * 70)

    try:
        from prototype.chroma_retrieval import ChromaDBRetriever

        logger.info(f"\nQuery: '{query}'")

        retriever = ChromaDBRetriever()

        # Check status first
        logger.info("\n📊 Checking collection status...")
        status = retriever.get_status()
        logger.info(f"   Documents in DB: {status.get('document_count', 0)}")

        if status.get("document_count", 0) == 0:
            logger.warning("   ⚠️  No documents in collection!")
            logger.warning("   Run 'python ingest.py' first")
            return

        # Retrieve documents
        logger.info("\n🔎 Retrieving documents...")
        docs = retriever.retrieve(query, k=3)

        logger.info(f"   Found: {len(docs)} documents\n")

        for i, doc in enumerate(docs, 1):
            title = doc.get("title", "Unknown")
            source = doc.get("source", "unknown")
            similarity = doc.get("similarity", "N/A")
            excerpt = doc.get("excerpt", "")

            logger.info(f"   [{i}] {title}")
            logger.info(f"       Source: {source}")
            if similarity != "N/A":
                logger.info(f"       Similarity: {similarity:.1%}")
            logger.info(f"       Excerpt: {excerpt[:100]}...")
            logger.info("")

        # Get stats
        logger.info("📈 Retrieval statistics:")
        stats = retriever.get_retrieval_stats(docs)
        logger.info(f"   Count: {stats['count']}")
        if stats["avg_similarity"] is not None:
            logger.info(f"   Avg Similarity: {stats['avg_similarity']:.1%}")
            logger.info(
                f"   Min/Max: {stats['min_similarity']:.1%} - {stats['max_similarity']:.1%}"
            )
        logger.info(f"   Sources: {stats['sources']}")

    except Exception as e:
        logger.error(f"✗ Retrieval test failed: {e}")
        raise


def run_integration_test():
    """Run full integration test with the RAG agent."""
    logger.info("\n" + "=" * 70)
    logger.info("🧪 INTEGRATION TEST (RAG Agent)")
    logger.info("=" * 70)

    try:
        from prototype.agents.retriever import RetrieverAgent
        from prototype.workflow.state import initialize_state

        # Create a test query
        test_query = "What is the Pythagorean theorem?"
        logger.info(f"\nTest query: '{test_query}'")

        # Initialize state
        state = initialize_state(test_query)

        # Create retriever agent
        logger.info("\n🤖 Initializing RetrieverAgent...")
        agent = RetrieverAgent()

        # Run retrieval
        logger.info("🔄 Running retrieval...")
        result_state = agent.run(state)

        # Check results
        docs = result_state.get("retrieved_docs", [])
        logger.info(f"✓ Retrieved {len(docs)} documents")

        for i, doc in enumerate(docs, 1):
            logger.info(
                f"  [{i}] {doc.get('title', '?')} (similarity: {doc.get('similarity', 'N/A')})"
            )

    except Exception as e:
        logger.error(f"✗ Integration test failed: {e}")
        raise


def print_usage():
    """Print usage instructions."""
    print(
        """
╔════════════════════════════════════════════════════════════════════════════╗
║                    ChromaDB RAG System - Quick Start                       ║
╚════════════════════════════════════════════════════════════════════════════╝

📋 SETUP STEPS:

1. Initialize ChromaDB:
   python setup_chromadb.py

2. Ingest documents:
   python ingest.py

3. Test retrieval:
   python setup_chromadb.py --test "What is x^2?"

4. Run full workflow:
   streamlit run app.py
   
   OR
   
   python main.py "your query"

📚 KEY FILES:

  chroma_setup.py      — ChromaDB client initialization
  ingest.py           — Document ingestion pipeline
  chroma_retrieval.py — High-level retrieval API
  db_updated.py       — Updated KBClient with ChromaDB support
  agents/retriever.py — RetrieverAgent (updated to use ChromaDB)

🔍 EXAMPLES:

  # Just setup and check status
  >>> from chroma_setup import get_chroma_client, get_chroma_collection
  >>> client = get_chroma_client()
  >>> collection = get_chroma_collection(client)
  >>> print(collection.count())
  
  # Query ChromaDB
  >>> from chroma_retrieval import ChromaDBRetriever
  >>> retriever = ChromaDBRetriever()
  >>> docs = retriever.retrieve("quadratic formula", k=5)
  >>> for d in docs:
  ...     print(d['title'], f"({d['similarity']:.1%})")

📊 FEATURES:

  ✓ Persistent storage (duckdb+parquet backend)
  ✓ Semantic search with embeddings
  ✓ Automatic chunking (700 char chunks with overlap)
  ✓ Fallback to JSONL/PDF if collection is empty
  ✓ Lazy model loading (efficient memory usage)
  ✓ Similarity scores for each result
  ✓ Query-focused excerpts

🚀 TROUBLESHOOTING:

  Q: "No documents in collection"
  A: Run 'python ingest.py' to load documents first
  
  Q: ChromaDB import error
  A: pip install chromadb>=0.4.0
  
  Q: sentence-transformers not found
  A: pip install sentence-transformers>=2.2.0
  
  Q: "Collection is empty" after ingestion
  A: Check that sample_docs.jsonl exists in prototype/data/

📞 NEED HELP?

  See MIGRATION_GUIDE.md for architecture overview
  Check README_LANGGRAPH.md for full system documentation
    """
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ChromaDB RAG Setup & Testing")
    parser.add_argument("--setup", action="store_true", help="Initialize ChromaDB")
    parser.add_argument("--ingest", action="store_true", help="Ingest documents")
    parser.add_argument(
        "--test",
        nargs="?",
        const="What is Pythag theorem?",
        help="Test retrieval with query",
    )
    parser.add_argument(
        "--integrate", action="store_true", help="Run integration test with RAG agent"
    )
    parser.add_argument("--all", action="store_true", help="Run full setup pipeline")
    parser.add_argument(
        "--help-usage", action="store_true", help="Show detailed usage examples"
    )

    args = parser.parse_args()

    if args.help_usage:
        print_usage()
        sys.exit(0)

    try:
        if args.all:
            # Full setup pipeline
            setup_chromadb()
            ingest_documents()
            test_retrieval("What is the Pythagorean theorem?")
            run_integration_test()

        else:
            if args.setup:
                setup_chromadb()

            if args.ingest:
                ingest_documents()

            if args.test:
                test_retrieval(args.test)

            if args.integrate:
                run_integration_test()

        if not any([args.setup, args.ingest, args.test, args.integrate, args.all]):
            # No args provided, show help
            print_usage()

    except Exception as e:
        logger.error(f"✗ Setup failed: {e}")
        sys.exit(1)
