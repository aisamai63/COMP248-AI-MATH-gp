"""
Chromadb integration: Document loading, retrieval, and persistence.

This module provides:
1. Fallback document loading (JSONL, PDFs)
2. ChromaDB-backed persistent retrieval
3. Embedding-based semantic search
4. Lazy model loading for efficiency
"""

import os
import json
from typing import List, Dict, Optional
import logging
from functools import lru_cache

try:
    import chromadb
    from chromadb.config import Settings

    CHROMADB_AVAILABLE = True
except Exception as e:
    CHROMADB_AVAILABLE = False
    logging.warning(f"Chromadb not available: {e}")

# PDF support
try:
    from pypdf import PdfReader

    PYPDF_AVAILABLE = True
except Exception as e:
    PYPDF_AVAILABLE = False
    logging.warning(f"pypdf not available: {e}")

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "sample_docs.jsonl")


def load_fallback_docs() -> List[Dict]:
    """
    Load fallback documents from JSONL and PDFs in the data directory.
    Returns:
        List of document dicts.
    """
    docs = []
    try:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            for line in f:
                docs.append(json.loads(line))
    except Exception as e:
        logging.error(f"Error loading JSONL docs: {e}")
    # Also load any PDFs in the data directory
    data_dir = os.path.dirname(DATA_PATH)
    for fname in os.listdir(data_dir):
        if fname.lower().endswith(".pdf") and PYPDF_AVAILABLE:
            path = os.path.join(data_dir, fname)
            try:
                reader = PdfReader(path)
                pages = []
                for p in reader.pages:
                    text = p.extract_text() or ""
                    pages.append(text.strip())
                full_text = "\n".join([p for p in pages if p])
                docs.append(
                    {
                        "id": os.path.splitext(fname)[0],
                        "title": os.path.splitext(fname)[0],
                        "source": "pdf",
                        "text": full_text,
                    }
                )
            except Exception as e:
                logging.warning(f"Could not read PDF {fname}: {e}")
    return docs


@lru_cache(maxsize=1)
def load_fallback_docs_cached() -> tuple:
    """
    Cached fallback document loader for faster Streamlit reruns.
    """
    return tuple(load_fallback_docs())


@lru_cache(maxsize=1)
def get_embedding_model():
    """
    Cached embedding model loader to avoid re-downloading/re-loading on each query.
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


def _extract_excerpt(text: str, query: str, window: int = 700) -> str:
    """
    Extract a short query-focused excerpt from a longer document.
    """
    if not text:
        return ""
    cleaned_text = " ".join(text.split())
    cleaned_query = " ".join(query.split()).strip().lower()
    if not cleaned_query:
        return cleaned_text[:window]

    lowered_text = cleaned_text.lower()
    query_index = lowered_text.find(cleaned_query)
    if query_index >= 0:
        start = max(0, query_index - window // 3)
        end = min(len(cleaned_text), query_index + len(cleaned_query) + window // 2)
        excerpt = cleaned_text[start:end]
    else:
        query_terms = [term for term in cleaned_query.split() if len(term) > 2]
        best_start = 0
        best_score = -1
        sentences = cleaned_text.split(".")
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            score = sum(1 for term in query_terms if term in sentence_lower)
            if score > best_score:
                best_score = score
                best_start = i
        excerpt = ". ".join(sentences[max(0, best_start - 1) : best_start + 2])
        if not excerpt:
            excerpt = cleaned_text[:window]

    excerpt = excerpt.strip()
    if len(excerpt) > window:
        excerpt = excerpt[:window].rsplit(" ", 1)[0] + "..."
    return excerpt


def _attach_excerpts(docs: List[Dict], query: str) -> List[Dict]:
    """
    Add compact excerpts to retrieved documents for answer generation.
    """
    prepared_docs: List[Dict] = []
    for doc in docs:
        prepared_doc = dict(doc)
        prepared_doc["excerpt"] = _extract_excerpt(doc.get("text", ""), query)
        prepared_docs.append(prepared_doc)
    return prepared_docs


class KBClient:
    """
    Knowledge Base client: handles document retrieval from persistent ChromaDB.

    Features:
    - Persistent vector storage using ChromaDB (duckdb+parquet backend)
    - Embedding-based semantic retrieval using SentenceTransformer
    - Automatic fallback to JSONL/PDF if ChromaDB collection is empty
    - Lazy loading of embedding model for efficiency
    - Similarity scores returned with results

    Usage:
        client = KBClient()
        docs = client.get_documents("What is the quadratic formula?", k=5)

        # Check ingestion status
        status = client.get_collection_status()
        print(f"Documents in DB: {status['document_count']}")
    """

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = "math_docs",
        use_chromadb: bool = True,
    ):
        """
        Initialize KB client with persistent ChromaDB backend.

        Args:
            persist_directory: Path to ChromaDB storage (default: .chroma_db/)
            collection_name: ChromaDB collection name (default: "math_docs")
            use_chromadb: If False, use fallback retrieval only
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.use_chromadb = use_chromadb
        self.client = None
        self.collection = None
        self.embedding_model = None

        if use_chromadb and CHROMADB_AVAILABLE:
            self._initialize_chroma()

    def _initialize_chroma(self):
        """Initialize ChromaDB client and collection (lazy init)."""
        try:
            from prototype.chroma_setup import get_chroma_client, get_chroma_collection

            self.client = get_chroma_client(persist_directory=self.persist_directory)
            self.collection = get_chroma_collection(
                self.client, collection_name=self.collection_name
            )
            logging.info(f"✓ ChromaDB KB client initialized")

        except Exception as e:
            logging.warning(f"Failed to initialize ChromaDB: {e}")
            self.client = None
            self.collection = None

    def _get_embedding_model(self):
        """Lazy load embedding model (only when needed)."""
        if self.embedding_model is None:
            try:
                from prototype.chroma_setup import get_embedding_model

                self.embedding_model = get_embedding_model()
            except Exception as e:
                logging.warning(f"Failed to load embedding model: {e}")
                return None
        return self.embedding_model

    def _check_collection_has_documents(self) -> bool:
        """Check if collection has any documents."""
        if self.collection is None:
            return False
        try:
            return self.collection.count() > 0
        except Exception:
            return False

    def get_documents(self, query: str, k: int = 5) -> List[Dict]:
        """
        Retrieve top-k relevant documents for a query.

        Retrieval priority:
        1. ChromaDB semantic search (if collection has documents)
        2. Fallback keyword-based search (on JSONL/PDFs)

        Args:
            query: User query string (will be embedded if using ChromaDB)
            k: Number of documents to retrieve (default: 5, max: 20)

        Returns:
            List of document dicts with:
            - id: Document/chunk ID
            - title: Document title
            - text: Full chunk text
            - excerpt: Query-focused excerpt
            - source: "chromadb" or "fallback"
            - similarity: (0-1) Only for ChromaDB results
            - metadata: Additional metadata dict
        """
        # Try ChromaDB first (if available and populated)
        if self.use_chromadb and self._check_collection_has_documents():
            try:
                return self._query_chromadb(query, k)
            except Exception as e:
                logging.warning(f"ChromaDB query failed, falling back: {e}")

        # Fallback to keyword-based retrieval
        return self._query_fallback(query, k)

    def _query_chromadb(self, query: str, k: int) -> List[Dict]:
        """
        Retrieve documents from ChromaDB using semantic similarity.

        Args:
            query: Query string (will be embedded)
            k: Number of results to return

        Returns:
            List of document dicts with similarity scores
        """
        # Get embedding model
        model = self._get_embedding_model()
        if model is None:
            raise RuntimeError("Embedding model not available")

        # Embed query
        query_embedding = model.encode([query])[0].tolist()

        # Query ChromaDB collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, 20),  # ChromaDB max is 20 results
            include=["documents", "metadatas", "distances"],
        )

        # Transform ChromaDB results to document format
        documents = []

        if results and results["documents"] and len(results["documents"]) > 0:
            docs = results["documents"][0]  # First query (we only sent 1)
            metadatas = results["metadatas"][0] if results["metadatas"] else []
            distances = results["distances"][0] if results["distances"] else []

            for i, doc_text in enumerate(docs):
                metadata = metadatas[i] if i < len(metadatas) else {}
                distance = distances[i] if i < len(distances) else 0.0

                # Convert distance to similarity (1 - distance for cosine)
                similarity = 1 - distance

                doc_dict = {
                    "id": metadata.get("original_id", f"doc_{i}"),
                    "title": metadata.get("title", "Unknown"),
                    "text": doc_text,
                    "source": metadata.get("source", "chromadb"),
                    "chunk_num": int(metadata.get("chunk_num", 0)),
                    "similarity": round(similarity, 4),
                    "metadata": metadata,
                }

                # Add excerpt (query-focused)
                doc_dict["excerpt"] = _extract_excerpt(doc_text, query)

                documents.append(doc_dict)

        if not documents:
            logging.warning(f"No results from ChromaDB for query: {query}")

        return documents

    def _query_fallback(self, query: str, k: int) -> List[Dict]:
        """
        Retrieve documents using keyword-based fallback (JSONL/PDFs).

        Used when:
        - ChromaDB is not available
        - ChromaDB collection is empty
        - ChromaDB query fails

        Args:
            query: Query string (will be matched against document text)
            k: Number of results to return

        Returns:
            List of document dicts
        """
        docs = list(load_fallback_docs_cached())

        if not query or not docs:
            return _attach_excerpts(docs[:k], query)

        # Score documents by query overlap
        q = query.lower()
        scored = []

        for d in docs:
            text = (d.get("title", "") + " " + d.get("text", "")).lower()
            # Simple term-frequency scoring
            score = text.count(q)
            scored.append((score, d))

        # Sort by relevance (highest first)
        scored.sort(key=lambda x: x[0], reverse=True)

        # Return top-k (prefer matches, fallback to all if none match)
        results = [d for s, d in scored if s > 0]
        if not results:
            results = docs

        return _attach_excerpts(results[:k], query)

    def get_collection_status(self) -> dict:
        """
        Get status of ChromaDB collection.

        Returns:
            Dict with:
            - status: "healthy", "empty", "unavailable", or "error"
            - document_count: Number of documents in collection
            - metadata: Collection metadata dict
        """
        if self.collection is None:
            return {
                "status": "unavailable",
                "message": "ChromaDB not initialized",
                "document_count": 0,
            }

        try:
            from prototype.chroma_setup import check_collection_status

            return check_collection_status(self.collection)
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "document_count": 0,
            }
