"""
ChromaDB Retrieval Utilities: Wrapper functions for the RAG agent.

This module provides high-level retrieval functions used by RetrieverAgent.
It handles:
- Persistent ChromaDB queries
- Automatic fallback logic
- Result formatting for LLM consumption

Usage:
    from chroma_retrieval import ChromaDBRetriever

    retriever = ChromaDBRetriever()
    docs = retriever.retrieve("What is x^2?", k=5)
"""

import logging
from typing import List, Dict, Optional
from prototype.config import db_config

logger = logging.getLogger(__name__)


class ChromaDBRetriever:
    """High-level ChromaDB retriever wrapper for RAG agents."""

    def __init__(
        self,
        use_chromadb: bool = True,
        collection_name: str = db_config.CHROMADB_COLLECTION_NAME,
    ):
        """
        Initialize retriever with ChromaDB support.

        Args:
            use_chromadb: Enable ChromaDB (default: True)
            collection_name: Collection name to query
        """
        self.use_chromadb = use_chromadb
        self.collection_name = collection_name
        self._kb_client = None

    @property
    def kb_client(self):
        """Lazy initialize KBClient on first access."""
        if self._kb_client is None:
            from prototype.db import KBClient

            self._kb_client = KBClient(
                collection_name=self.collection_name,
                use_chromadb=self.use_chromadb,
            )
        return self._kb_client

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """
        Retrieve top-k documents for a query.

        Args:
            query: User query string
            k: Number of documents to retrieve

        Returns:
            List of document dicts with metadata
        """
        return self.kb_client.get_documents(query, k=k)

    def get_status(self) -> dict:
        """Get collection status and document count."""
        return self.kb_client.get_collection_status()

    def format_for_llm(self, docs: List[Dict]) -> str:
        """
        Format retrieved documents for LLM consumption.

        Args:
            docs: List of document dicts from retrieve()

        Returns:
            Formatted string with document excerpts
        """
        if not docs:
            return "No relevant documents found."

        formatted = "Retrieved documents:\n\n"

        for i, doc in enumerate(docs, 1):
            excerpt = doc.get("excerpt", "")
            title = doc.get("title", "Unknown")
            source = doc.get("source", "unknown")
            similarity = doc.get("similarity", None)

            formatted += f"[{i}] {title} (from {source})\n"

            if similarity is not None:
                formatted += f"    Similarity: {similarity:.2%}\n"

            formatted += f"    {excerpt}\n\n"

        return formatted

    def get_retrieval_stats(self, docs: List[Dict]) -> dict:
        """
        Compute retrieval statistics for logging/monitoring.

        Args:
            docs: List of documents retrieved

        Returns:
            Dict with stats (count, avg similarity, sources, etc)
        """
        stats = {
            "count": len(docs),
            "sources": {},
            "avg_similarity": None,
            "min_similarity": None,
            "max_similarity": None,
        }

        if not docs:
            return stats

        similarities = []
        for doc in docs:
            source = doc.get("source", "unknown")
            stats["sources"][source] = stats["sources"].get(source, 0) + 1

            similarity = doc.get("similarity")
            if similarity is not None:
                similarities.append(similarity)

        if similarities:
            stats["avg_similarity"] = sum(similarities) / len(similarities)
            stats["min_similarity"] = min(similarities)
            stats["max_similarity"] = max(similarities)

        return stats


if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(level=logging.INFO)

    retriever = ChromaDBRetriever()

    # Check collection status
    status = retriever.get_status()
    print(f"Collection status: {status}")
    print(f"Documents in DB: {status.get('document_count', 0)}\n")

    # Test retrieval
    query = "What is the Pythagorean theorem?"
    docs = retriever.retrieve(query, k=3)

    print(f"Query: {query}")
    print(f"Results: {len(docs)} documents\n")

    # Format for LLM
    formatted = retriever.format_for_llm(docs)
    print(formatted)

    # Get stats
    stats = retriever.get_retrieval_stats(docs)
    print(f"Retrieval stats: {stats}")
