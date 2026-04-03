"""
Summarizer Agent: Generates summaries using LLM.

Role:
  - Receives query + documents
  - Uses LLM to generate focused summary
  - Returns compact answer

Flow:
  Input: user_query, retrieved_docs
  Step 1: Build prompt with query + doc excerpts
  Step 2: Call Mistral LLM
  Step 3: Return summary
  Output: summary
"""

import logging
import time
from typing import Optional
from prototype.agents.base import BaseAgent
from prototype.workflow.state import WorkflowState
from prototype.config import llm_config

logger = logging.getLogger(__name__)


class SummarizerAgent(BaseAgent):
    """
    Summarizer agent: generates LLM-based summaries of retrieved documents.

    Uses Mistral API to create concise, query-focused summaries.
    Falls back to naive summarization if LLM is unavailable.
    """

    name = "Summarizer"

    def __init__(self, use_llm: bool = True):
        """
        Initialize Summarizer agent.

        Args:
            use_llm: Whether to use LLM (True) or fallback (False)
        """
        super().__init__()
        self.use_llm = use_llm
        self.llm_ready = False
        self.mistral_client = None

        if self.use_llm and llm_config.MISTRAL_API_KEY:
            try:
                from mistralai.client import Mistral

                self.mistral_client = Mistral(api_key=llm_config.MISTRAL_API_KEY)
                self.llm_ready = True
                self.logger.info("Mistral client initialized successfully")
            except Exception as e:
                self.logger.warning(
                    f"Failed to initialize Mistral: {e}. Using fallback."
                )
                self.llm_ready = False

    def run(self, state: WorkflowState) -> WorkflowState:
        """
        Execute summarization: generate summary from documents.

        Args:
            state: Workflow state with retrieved_docs and user_query

        Returns:
            Modified state with summary populated
        """
        try:
            start_time = time.time()

            docs = state["retrieved_docs"]
            query = state["user_query"]

            if not docs:
                state["summary"] = "[No documents retrieved; cannot generate summary]"
                self.logger.warning("No documents to summarize")
                return state

            self._log_step(
                "Generating summary", {"num_docs": len(docs), "query": query}
            )

            # Choose summarization method
            if self.use_llm and self.llm_ready:
                summary = self._summarize_with_llm(query, docs)
            else:
                self.logger.info("Using fallback summarization (no LLM)")
                summary = self._summarize_fallback(query, docs)

            state["summary"] = summary

            # Log execution time
            elapsed = time.time() - start_time
            state["metadata"]["timestamps"]["summarizer"] = elapsed

            self._log_step("Summary generated", {"length": len(summary.split())})

            return state

        except Exception as e:
            return self._handle_error(state, e)

    def _summarize_with_llm(self, query: str, docs: list) -> str:
        """
        Summarize using Mistral LLM.

        Args:
            query: User query
            docs: Retrieved documents

        Returns:
            LLM-generated summary
        """
        try:
            prompt = self._build_summary_prompt(query, docs)

            response = self.mistral_client.chat.complete(
                model=llm_config.MISTRAL_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=llm_config.MISTRAL_MAX_TOKENS_DEFAULT,
                temperature=llm_config.MISTRAL_TEMPERATURE,
            )

            content = response.choices[0].message.content
            return content.strip()

        except Exception as e:
            self.logger.error(f"LLM summarization failed: {e}")
            return f"[Summarization error: {str(e)}]"

    def _summarize_fallback(self, query: str, docs: list) -> str:
        """
        Fallback: simple summarization without LLM.

        Extracts first sentence from each document and joins them.

        Args:
            query: User query (unused in fallback)
            docs: Retrieved documents

        Returns:
            Simple concatenated summary
        """
        parts = []
        for doc in docs:
            title = doc.get("title", "Unknown")
            text = doc.get("excerpt") or doc.get("text", "")

            # Get first sentence
            first_sentence = text.split(".")[0] if text else ""
            if first_sentence:
                parts.append(f"**{title}**: {first_sentence.strip()}.")

        return " ".join(parts) if parts else "[No content available]"

    def _build_summary_prompt(self, query: str, docs: list) -> str:
        """
        Build prompt for LLM summarization.

        Instructs the LLM to:
        - Answer the user's question
        - Use provided excerpts only
        - Structure response in 3 sections: answer, bullet summary, related points

        Args:
            query: User query
            docs: Retrieved documents

        Returns:
            Formatted prompt for LLM
        """
        prompt = (
            "You are an expert research assistant. Answer the user's question using ONLY "
            "the provided document excerpts. Do not use external knowledge.\n\n"
            f"USER QUESTION: {query}\n\n"
            "RESPONSE FORMAT:\n"
            "1. Direct Answer (1-2 sentences)\n"
            "2. Key Points (3-5 bullet points)\n"
            "3. Related Concepts (2-3 items if applicable)\n\n"
            "DOCUMENTS:\n"
        )

        for i, doc in enumerate(docs, 1):
            excerpt = doc.get("excerpt") or doc.get("text", "")
            title = doc.get("title", "Document")
            source = doc.get("source", "unknown")
            prompt += f"\n[Document {i}] {title} (source: {source})\n{excerpt}\n"

        prompt += "\nPlease answer based only on the above documents."
        return prompt
