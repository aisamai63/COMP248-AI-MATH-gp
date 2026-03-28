"""Simple agent prototypes: Planner, SearchAgent, SummarizerAgent, ReflectiveAgent."""

import os
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

from prototype import db

OPENAI_KEY = os.getenv("OPENAI_API_KEY")


class Planner:
    def __init__(self, kb_client):
        self.kb = kb_client

    def handle_query(self, query: str) -> Dict:
        docs = self.kb.get_documents(query, k=5)
        return {"query": query, "docs": docs}


class SearchAgent:
    def __init__(self, kb_client):
        self.kb = kb_client

    def retrieve(self, query: str, k: int = 5):
        return self.kb.get_documents(query, k=k)


class SummarizerAgent:
    def __init__(self):
        pass

    def summarize(self, docs: List[Dict]) -> str:
        # Naive summarization for demo: join titles and first sentences
        parts = []
        for d in docs:
            text = d.get("text", "")
            parts.append((d.get("title", ""), text.split(".")[0]))
        summary = "; ".join([f"{t}: {s}." for t, s in parts])
        return summary


class ReflectiveAgent:
    def __init__(self):
        pass

    def reflect(self, summary: str, docs: List[Dict]) -> Dict:
        # Simple heuristics: length, doc_coverage
        coverage = min(100, int(len(summary) / 2))
        return {"confidence": coverage, "notes": "Auto reflection (demo)"}
