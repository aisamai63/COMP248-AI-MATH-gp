"""Chromadb helper with fallback to local JSONL for demo."""

import os
import json
from typing import List, Dict

try:
    import chromadb
    from chromadb.config import Settings

    CHROMADB_AVAILABLE = True
except Exception:
    CHROMADB_AVAILABLE = False

# PDF support
try:
    from pypdf import PdfReader

    PYPDF_AVAILABLE = True
except Exception:
    PYPDF_AVAILABLE = False

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "sample_docs.jsonl")


def load_fallback_docs() -> List[Dict]:
    docs = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
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
            except Exception:
                # ignore unreadable PDFs for demo
                continue
    return docs


class KBClient:
    def __init__(self, persist_directory: str = None):
        self.persist_directory = persist_directory
        self.client = None
        if CHROMADB_AVAILABLE:
            try:
                self.client = chromadb.Client()
            except Exception:
                self.client = None

    def get_documents(self, query: str, k: int = 5):
        if self.client is None:
            docs = load_fallback_docs()
            # simple substring filter to surface relevant PDFs/docs for demo
            if not query:
                return docs[:k]
            q = query.lower()
            scored = []
            for d in docs:
                text = (d.get("title", "") + " " + d.get("text", "")).lower()
                score = text.count(q)
                scored.append((score, d))
            # return non-zero matches first, else top-k
            scored.sort(key=lambda x: x[0], reverse=True)
            results = [d for s, d in scored if s > 0]
            if not results:
                return docs[:k]
            return results[:k]
        # naive: return all metadata for demo
        # Real implementation would perform embedding + similarity search
        return load_fallback_docs()
