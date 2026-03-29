"""
Chromadb helper with fallback to local JSONL for demo.
Handles document loading, retrieval, and PDF support.
"""

import os
import json
from typing import List, Dict, Optional
import logging

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


class KBClient:
    """
    Knowledge Base client: handles document retrieval from Chromadb or fallback.
    """

    def __init__(self, persist_directory: Optional[str] = None):
        self.persist_directory = persist_directory
        self.client = None
        if CHROMADB_AVAILABLE:
            try:
                self.client = chromadb.Client()
            except Exception as e:
                logging.warning(f"Failed to initialize Chromadb client: {e}")
                self.client = None

    def get_documents(self, query: str, k: int = 5) -> List[Dict]:
        """
        Retrieve top-k relevant documents for a query using Chromadb if available, else fallback.
        Args:
            query: Query string.
            k: Number of documents to retrieve.
        Returns:
            List of document dicts.
        """
        if self.client is None:
            docs = load_fallback_docs()
            # Simple substring filter to surface relevant PDFs/docs for demo
            if not query:
                return docs[:k]
            q = query.lower()
            scored = []
            for d in docs:
                text = (d.get("title", "") + " " + d.get("text", "")).lower()
                score = text.count(q)
                scored.append((score, d))
            # Return non-zero matches first, else top-k
            scored.sort(key=lambda x: x[0], reverse=True)
            results = [d for s, d in scored if s > 0]
            if not results:
                return docs[:k]
            return results[:k]
        # Chromadb available: perform embedding-based retrieval
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
        except ImportError:
            # Fallback if dependencies missing
            return load_fallback_docs()[:k]

        # Load all docs (for demo, in real use, store embeddings in Chromadb)
        docs = load_fallback_docs()
        texts = [d.get("text", "") for d in docs]
        # Compute embeddings
        model = SentenceTransformer("all-MiniLM-L6-v2")
        doc_embs = model.encode(texts)
        query_emb = model.encode([query])[0]
        # Compute cosine similarity
        sims = np.dot(doc_embs, query_emb) / (
            np.linalg.norm(doc_embs, axis=1) * np.linalg.norm(query_emb) + 1e-8
        )
        # Rank and return top-k
        ranked = sorted(zip(sims, docs), key=lambda x: x[0], reverse=True)
        return [d for _, d in ranked[:k]]
