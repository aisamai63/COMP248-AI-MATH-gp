"""
Agent module for Math Inquiries prototype.
Includes: Planner, SearchAgent, SummarizerAgent, ReflectiveAgent, ToolAgent.
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

from prototype import db


# API keys for LLMs
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")


class Planner:
    """
    Planner agent: routes queries to workers and composes tasks.
    """

    def __init__(self, kb_client: Any):
        """
        Args:
            kb_client: Knowledge base client instance.
        """
        self.kb = kb_client

    def handle_query(self, query: str) -> Dict[str, Any]:
        """
        Handles a user query by retrieving relevant documents.
        Args:
            query: User's query string.
        Returns:
            Dict with query and retrieved docs.
        """
        docs = self.kb.get_documents(query, k=5)
        return {"query": query, "docs": docs}


class SearchAgent:
    """
    SearchAgent: retrieves relevant documents from the knowledge base.
    """

    def __init__(self, kb_client: Any):
        self.kb = kb_client

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top-k relevant documents for a query.
        Args:
            query: Query string.
            k: Number of documents to retrieve.
        Returns:
            List of document dicts.
        """
        return self.kb.get_documents(query, k=k)


class SummarizerAgent:
    """
    SummarizerAgent: summarizes a list of documents using OpenAI, Gemini, or fallback logic.
    """

    def __init__(self, use_llm: str = None, max_tokens: int = 120):
        """
        Args:
            use_llm: 'openai', 'gemini', or None
            max_tokens: max tokens for summary
        """
        self.use_llm = use_llm  # 'openai', 'gemini', or None
        self.llm_ready = False
        self.max_tokens = max_tokens
        self.openai_client = None
        self.gemini_model = None

        if self.use_llm == "openai" and OPENAI_KEY:
            try:
                from openai import OpenAI

                self.openai_client = OpenAI(api_key=OPENAI_KEY)
                self.llm_ready = True
            except Exception:
                self.llm_ready = False
        elif self.use_llm == "gemini" and GEMINI_KEY:
            try:
                import google.generativeai as genai

                genai.configure(api_key=GEMINI_KEY)
                self.gemini_model = genai.GenerativeModel("gemini-pro")
                self.llm_ready = True
            except Exception:
                self.llm_ready = False

    def summarize(
        self, docs: List[Dict[str, Any]], max_tokens: Optional[int] = None
    ) -> str:
        """
        Summarize a list of documents using OpenAI, Gemini, or fallback.
        Args:
            docs: List of document dicts.
            max_tokens: override default max tokens
        Returns:
            Summary string.
        """
        if self.use_llm == "openai" and self.llm_ready and docs:
            prompt = """Summarize the following documents for a research analyst.\n\n"""
            for i, d in enumerate(docs, 1):
                prompt += f"Document {i} (Title: {d.get('title', '')}):\n{d.get('text', '')}\n\n"
            prompt += "\nSummary:"
            try:
                tokens = max_tokens if max_tokens is not None else self.max_tokens
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=tokens,
                    temperature=0.4,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                return f"[LLM summarization failed: {e}]"
        elif self.use_llm == "gemini" and self.llm_ready and docs:
            prompt = """Summarize the following documents for a research analyst.\n\n"""
            for i, d in enumerate(docs, 1):
                prompt += f"Document {i} (Title: {d.get('title', '')}):\n{d.get('text', '')}\n\n"
            prompt += "\nSummary:"
            try:
                tokens = max_tokens if max_tokens is not None else self.max_tokens
                response = self.gemini_model.generate_content(
                    prompt,
                    generation_config={"max_output_tokens": tokens, "temperature": 0.4},
                )
                return response.text.strip()
            except Exception as e:
                return f"[Gemini summarization failed: {e}]"
        # Naive summarization: join titles and first sentences
        parts = []
        for d in docs:
            text = d.get("text", "")
            first_sentence = text.split(".")[0] if text else ""
            parts.append((d.get("title", ""), first_sentence))
        summary = "; ".join([f"{t}: {s}." for t, s in parts])
        return summary


class ReflectiveAgent:
    """
    ReflectiveAgent: evaluates summary quality and suggests improvements.
    """

    def __init__(self):
        pass

    def reflect(self, summary: str, docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Reflect on the summary using simple heuristics.
        Args:
            summary: The generated summary string.
            docs: List of source documents.
        Returns:
            Dict with confidence score and notes.
        """
        # Heuristics: coverage (summary length vs. docs), redundancy, length, factuality (placeholder)
        num_docs = len(docs)
        summary_len = len(summary.split())
        avg_doc_len = (
            sum(len(d.get("text", "").split()) for d in docs) / num_docs
            if num_docs
            else 1
        )
        coverage = (
            min(100, int((summary_len / (avg_doc_len * num_docs)) * 100))
            if num_docs
            else 0
        )
        redundancy = (
            max(0, 100 - int(len(set(summary.split())) / (summary_len + 1) * 100))
            if summary_len
            else 0
        )
        notes = []
        if coverage < 50:
            notes.append("Low coverage: consider retrieving more documents.")
        if redundancy > 50:
            notes.append("High redundancy: summary may be repetitive.")
        if summary_len < 30:
            notes.append("Summary is short; may lack detail.")
        return {
            "confidence": coverage,
            "redundancy": redundancy,
            "summary_length": summary_len,
            "notes": notes or ["Auto reflection (demo)"],
        }


class ToolAgent:
    """
    ToolAgent: interfaces with external tools/APIs (stub for demo).
    """

    def __init__(self):
        pass

    def use_tool(self, tool_name: str, input_data: Any) -> Any:
        """
        Stub for tool use (e.g., web search, calculator).
        Args:
            tool_name: Name of the tool.
            input_data: Input for the tool.
        Returns:
            Tool output (stub).
        """
        return f"Tool '{tool_name}' called with input: {input_data} (stub)"
