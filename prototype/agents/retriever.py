"""
Retriever Agent: Performs RAG (Retrieval-Augmented Generation).

Role:
  - Receives user query
  - Searches knowledge base
  - Returns top-k relevant documents

Flow:
  Input: user_query, k (from Planner)
  Step 1: Embed query
  Step 2: Search KB
  Step 3: Extract excerpts
  Output: retrieved_docs
"""

import logging
import time
from typing import List, Dict, Any
from prototype.agents.base import BaseAgent
from prototype.workflow.state import WorkflowState
from prototype.chroma_retrieval import ChromaDBRetriever
from prototype.config import rag_config

logger = logging.getLogger(__name__)


class RetrieverAgent(BaseAgent):
    """
    Retriever agent: performs embedding-based document retrieval.

    This agent wraps the existing KBClient and integrates it into the LangGraph.
    It retrieves top-k documents most similar to the user query using
    embeddings and cosine similarity.
    """

    name = "Retriever"

    def __init__(self, retriever: ChromaDBRetriever = None):
        """
        Initialize the Retriever agent with ChromaDB support.

        Args:
            retriever: ChromaDBRetriever instance (for dependency injection)
                      If None, creates a new one

        Features:
        - Persistent ChromaDB storage (not in-memory)
        - Semantic similarity search using embeddings
        - Automatic fallback to JSONL/PDF retrieval
        """
        super().__init__()
        self.retriever = retriever or ChromaDBRetriever()

    def run(self, state: WorkflowState) -> WorkflowState:
        """
        Execute retrieval: search KB and get top-k documents.

        Args:
            state: Workflow state with user_query

        Returns:
            Modified state with retrieved_docs populated

        Flow:
          1. Extract k from state metadata or use default
          2. Call KB client to retrieve documents
          3. Log what we retrieved
          4. Return updated state
        """
        try:
            start_time = time.time()

            query = state["user_query"]
            k = (
                state["metadata"]
                .get("decisions", {})
                .get("planner", {})
                .get("k_documents", rag_config.K_DEFAULT)
            )

            self._log_step("Retrieving documents", {"query": query, "k": k})

            # Retrieve documents from persistent ChromaDB
            docs = self.retriever.retrieve(query, k=k)

            self._log_step(
                "Retrieval complete",
                {
                    "num_docs": len(docs),
                    "docs_titles": [d.get("title", "?") for d in docs],
                },
            )

            # Update state
            state["retrieved_docs"] = docs

            # Log execution time
            elapsed = time.time() - start_time
            state["metadata"]["timestamps"]["retriever"] = elapsed

            return state

        except Exception as e:
            return self._handle_error(state, e)

    def get_retrieval_stats(self, state: WorkflowState) -> Dict[str, Any]:
        """
        Return statistics about retrieval for UI display.

        Args:
            state: Workflow state

        Returns:
            Dict with retrieval stats
        """
        docs = state["retrieved_docs"]
        return {
            "num_retrieved": len(docs),
            "sources": [d.get("source", "unknown") for d in docs],
            "average_length": (
                sum(len(d.get("text", "").split()) for d in docs) / len(docs)
                if docs
                else 0
            ),
        }
