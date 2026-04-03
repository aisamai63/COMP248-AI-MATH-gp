"""
Planner Agent: Routes queries and makes high-level decisions.

Role:
  - Receives user query
  - Decides workflow path (which agents to invoke)
  - Manages iteration logic (when to retrieve more docs)
  - Coordinates overall workflow

Flow:
  Input: user_query
  Step 1: Classify query type (e.g., "definition", "calculation", "proof")
  Step 2: Route to appropriate agents
  Step 3: Check if re-retrieval needed based on reflection feedback
  Output: Updated state + decision for next step
"""

import logging
import re
from typing import Dict, Any, Tuple
from prototype.agents.base import BaseAgent
from prototype.workflow.state import WorkflowState
from prototype.config import reflection_config, rag_config, runtime_config

logger = logging.getLogger(__name__)


class PlannerAgent(BaseAgent):
    """
    Planner agent: orchestrates workflow and makes routing decisions.

    Responsibilities:
    - Classify query intent
    - Route to SearchAgent for retrieval
    - Route to SummarizerAgent for summarization
    - Route to ReflectiveAgent for evaluation
    - Use reflection feedback to decide re-retrieval

    Design Decision:
      We keep the Planner in LangGraph as a decision-making node
      rather than implementing it as a separate agent. This is cleaner
      for control flow (use conditional_edges for routing).
    """

    name = "Planner"

    def __init__(self):
        super().__init__()
        self.query_type_keywords = {
            "definition": ["define", "what is", "explain", "meaning"],
            "calculation": ["calculate", "compute", "solve", "evaluate", "factor"],
            "proof": ["prove", "show", "derive", "demonstrate"],
            "comparison": ["compare", "difference", "versus", "vs"],
        }
        self.web_search_keywords = [
            "latest",
            "recent",
            "current",
            "today",
            "news",
            "update",
            "2025",
            "2026",
        ]

    def run(self, state: WorkflowState) -> WorkflowState:
        """
        Execute Planner logic: classify query and decide workflow path.

        Args:
            state: Workflow state

        Returns:
            Modified state with routing decision
        """
        try:
            self._log_step("Starting planning phase", {"query": state["user_query"]})

            # Step 1: Classify query type
            query_type = self._classify_query(state["user_query"])
            self._log_step("Query classified", {"type": query_type})

            # Step 2: Update iteration count
            state["iteration_count"] += 1

            # Step 3: Evaluate reflection feedback and decide loop action.
            confidence, next_action = self._evaluate_feedback(state)
            state["should_continue"] = next_action == "retry"

            # Step 4: Decide retrieval parameters for this iteration.
            k_docs = self._decide_retrieval_k(state)
            self._log_step(
                "Decided retrieval K",
                {
                    "k": k_docs,
                    "iteration": state["iteration_count"],
                    "confidence": confidence,
                    "next_action": next_action,
                },
            )

            # Step 5: Decide tool hints for ToolAgent.
            selected_tools = self._decide_tools(state["user_query"], query_type)
            self._log_step("Decided tool hints", {"tools": selected_tools})

            # Step 6: Store current planner decision in metadata.
            planner_decision = {
                "query_type": query_type,
                "k_documents": k_docs,
                "iteration": state["iteration_count"],
                "tools": selected_tools,
                "confidence": confidence,
                "next_action": next_action,
            }
            state["metadata"]["decisions"]["planner"] = planner_decision

            # Keep history for traceability across iterative loops.
            planner_history = state["metadata"]["decisions"].setdefault(
                "planner_history", []
            )
            planner_history.append(planner_decision)

            trace = state["metadata"].setdefault("trace", [])
            trace.append(
                {
                    "node": "planner",
                    "iteration": state["iteration_count"],
                    "confidence": confidence,
                    "k_documents": k_docs,
                    "next_action": next_action,
                }
            )

            # Next step will be retrieval (handled via graph edges)
            return state

        except Exception as e:
            return self._handle_error(state, e)

    def _classify_query(self, query: str) -> str:
        """
        Classify query type based on keywords.

        Args:
            query: User's query

        Returns:
            Query type string

        Note:
          In production, this could use an LLM for more sophisticated classification.
          For now, we use simple keyword matching.
        """
        query_lower = query.lower()

        for query_type, keywords in self.query_type_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return query_type

        # Default to "research"
        return "research"

    def _decide_retrieval_k(self, state: WorkflowState) -> int:
        """
        Decide how many documents to retrieve.

        Strategy:
        - First iteration: Use default k (usually 5)
        - After low confidence: Expand k (usually 10)
        - Further iterations: Keep expanding (usually 15, 20)

        Args:
            state: Workflow state (contains iteration_count and reflection_metrics)

        Returns:
            k value for retrieval
        """
        iteration = state["iteration_count"]

        if runtime_config.FAST_MODE:
            return min(3, rag_config.K_DEFAULT)

        # First pass uses default breadth.
        if iteration <= 1:
            return rag_config.K_DEFAULT

        confidence = state.get("reflection_metrics", {}).get("confidence", 1.0)
        if confidence < reflection_config.CONFIDENCE_THRESHOLD:
            # Iterative recovery plan: expand context on low-confidence loops.
            # Iter 2 starts at K_EXPANDED, then grows by K_DEFAULT until K_MAX.
            growth = (iteration - 2) * rag_config.K_DEFAULT
            return min(rag_config.K_EXPANDED + growth, rag_config.K_MAX)

        return rag_config.K_DEFAULT

    def _evaluate_feedback(self, state: WorkflowState) -> Tuple[float, str]:
        """Evaluate reflection confidence and choose retry/continue action."""
        metrics = state.get("reflection_metrics", {})
        iteration = state["iteration_count"]

        if not metrics:
            # Initial pass: no feedback yet, proceed with first retrieval.
            return 1.0, "retry"

        confidence = metrics.get("confidence", 0.0)

        if confidence >= reflection_config.CONFIDENCE_THRESHOLD:
            return confidence, "continue"

        if iteration >= reflection_config.MAX_ITERATIONS:
            return confidence, "continue"

        return confidence, "retry"

    def _decide_tools(self, query: str, query_type: str) -> list:
        """Return planner tool hints consumed by ToolAgent."""
        tools = []
        query_lower = query.lower()

        equation_like = bool(
            re.search(r"[a-z].*=|=.*[a-z]|\d+[a-z]|[a-z]\d+", query_lower)
        )
        matrix_like = any(
            tok in query_lower for tok in ["matrix", "row echelon", "rref", "ref"]
        )

        if query_type == "calculation" or equation_like or matrix_like:
            tools.append("calculator")

        if any(keyword in query_lower for keyword in self.web_search_keywords):
            tools.append("web_search")

        return tools

    def should_continue(self, state: WorkflowState) -> bool:
        """
        Determine if workflow should continue or terminate.

        Stopping conditions:
        1. User feedback satisfied (high confidence)
        2. Max iterations reached
        3. Error occurred

        Args:
            state: Current workflow state

        Returns:
            True if workflow should continue
        """
        if not state["should_continue"]:
            return False

        if state["error_message"]:
            self.logger.warning(f"Error detected: {state['error_message']}")
            return False

        # If no metrics yet, continue (first iteration)
        if not state["reflection_metrics"]:
            return True

        confidence = state["reflection_metrics"].get("confidence", 0.0)
        iteration = state["iteration_count"]

        # Stop if confident OR max iterations reached
        if confidence >= reflection_config.CONFIDENCE_THRESHOLD:
            self._log_step("Confidence threshold reached", {"confidence": confidence})
            return False

        if iteration >= reflection_config.MAX_ITERATIONS:
            self._log_step("Max iterations reached", {"iteration": iteration})
            return False

        # Continue to next iteration
        return True
