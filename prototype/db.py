"""
Chromadb helper with fallback to local JSONL for demo.
Handles document loading, retrieval, and PDF support.
"""

import os
import json
from typing import List, Dict, Optional
import logging
from functools import lru_cache
from prototype.config import rag_config, excerpt_config, db_config

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

    return SentenceTransformer(rag_config.EMBEDDING_MODEL)


def _extract_excerpt(
    text: str,
    query: str,
    window: int = excerpt_config.EXCERPT_WINDOW_CHARS,
) -> str:
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
    """Knowledge Base client with persistent ChromaDB retrieval and safe fallback."""

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = db_config.CHROMADB_COLLECTION_NAME,
        use_chromadb: bool = True,
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.use_chromadb = use_chromadb
        self.client = None
        self.collection = None
        self.embedding_model = None

        if use_chromadb and CHROMADB_AVAILABLE:
            self._initialize_chroma()

    def _initialize_chroma(self):
        """Initialize persistent Chroma client and collection."""
        try:
            from prototype.chroma_setup import get_chroma_client, get_chroma_collection

            self.client = get_chroma_client(persist_directory=self.persist_directory)
            self.collection = get_chroma_collection(
                self.client,
                collection_name=self.collection_name,
                distance_metric=db_config.CHROMADB_DISTANCE_METRIC,
            )
            logging.info("ChromaDB KBClient initialized")
        except Exception as e:
            logging.warning(f"Failed to initialize ChromaDB client: {e}")
            self.client = None
            self.collection = None

    def _get_embedding_model(self):
        if self.embedding_model is None:
            self.embedding_model = get_embedding_model()
        return self.embedding_model

    def _has_indexed_documents(self) -> bool:
        if self.collection is None:
            return False
        try:
            return self.collection.count() > 0
        except Exception:
            return False

    def _query_chromadb(self, query: str, k: int) -> List[Dict]:
        model = self._get_embedding_model()
        query_embedding = model.encode([query])[0].tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, rag_config.K_MAX),
            include=["documents", "metadatas", "distances"],
        )

        documents: List[Dict] = []
        docs = results.get("documents", [[]])[0] if results else []
        metadatas = results.get("metadatas", [[]])[0] if results else []
        distances = results.get("distances", [[]])[0] if results else []

        for i, doc_text in enumerate(docs):
            metadata = metadatas[i] if i < len(metadatas) else {}
            distance = distances[i] if i < len(distances) else 1.0
            similarity = max(0.0, min(1.0, 1 - float(distance)))

            prepared = {
                "id": metadata.get("original_id", f"doc_{i}"),
                "title": metadata.get("title", "Unknown"),
                "source": metadata.get("source", "chromadb"),
                "text": doc_text,
                "chunk_num": int(metadata.get("chunk_num", 0)),
                "similarity": round(similarity, 4),
                "metadata": metadata,
                "excerpt": _extract_excerpt(doc_text, query),
            }
            documents.append(prepared)

        return documents

    def _query_fallback(self, query: str, k: int) -> List[Dict]:
        docs = list(load_fallback_docs_cached())
        docs = _attach_excerpts(docs, query)
        if not query:
            return docs[:k]

        q = query.lower()
        scored = []
        for d in docs:
            text = (d.get("title", "") + " " + d.get("text", "")).lower()
            score = text.count(q)
            scored.append((score, d))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [d for s, d in scored if s > 0]
        if not results:
            results = docs
        return results[:k]

    def get_documents(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve top-k documents using persistent ChromaDB, fallback when needed."""
        if self.use_chromadb and self._has_indexed_documents():
            try:
                chroma_docs = self._query_chromadb(query, k)
                if chroma_docs:
                    return chroma_docs
            except Exception as e:
                logging.warning(f"ChromaDB query failed, using fallback: {e}")

        return self._query_fallback(query, k)

    def get_collection_status(self) -> dict:
        """Return collection diagnostics for UI/CLI reporting."""
        if self.collection is None:
            return {
                "status": "unavailable",
                "document_count": 0,
                "collection_name": self.collection_name,
            }
        try:
            return {
                "status": "healthy" if self.collection.count() > 0 else "empty",
                "document_count": self.collection.count(),
                "collection_name": self.collection_name,
                "metadata": self.collection.metadata,
            }
        except Exception as e:
            return {
                "status": "error",
                "document_count": 0,
                "collection_name": self.collection_name,
                "error": str(e),
            }
