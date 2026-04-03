"""
LangGraph workflow: defines the multi-agent execution graph.

Architecture Overview:

    User Query
       |
       v
    [Planner]  ← Classify query & decide retrieval params
       |
       v
    [Retriever]  ← Search KB for top-k docs
       |
       v
    [Summarizer]  ← Generate LLM summary
       |
       v
    [Reflector]  ← Evaluate quality & decide continue
       |
    +-- if confidence < threshold --> back to [Planner] (adjust plan, then re-retrieve)
       |
       +-- if confidence >= threshold or max_iterations --> to [ToolAgent]
       |
       v
    [ToolAgent] (optional)  ← Execute external tools if needed
       |
       v
    Return Final Response

Control Flow:
- Nodes: Each agent is a node
- Edges: Sequential execution
- Conditional edges: Re-retrieval loop based on confidence

Key Design Decisions:
1. State is shared and passed through all nodes
2. Each agent modifies state in-place and returns it
3. Conditional edge from Reflector decides planner re-entry
4. ToolAgent runs at end (optional based on query needs)
"""

import logging
import time
from typing import Literal
from langgraph.graph import StateGraph, END
from prototype.workflow.state import WorkflowState, initialize_state
from prototype.agents import (
    PlannerAgent,
    RetrieverAgent,
    SummarizerAgent,
    ReflectiveAgent,
    ToolAgent,
)
from prototype.chroma_retrieval import ChromaDBRetriever
from prototype.config import reflection_config, runtime_config

logger = logging.getLogger(__name__)


def _ensure_metadata(state: WorkflowState) -> None:
    """Ensure metadata containers exist before instrumentation updates."""
    metadata = state.setdefault("metadata", {})
    metadata.setdefault("agent_sequence", [])
    metadata.setdefault("timestamps", {})
    metadata.setdefault("decisions", {})


def _record_node_metadata(
    state: WorkflowState,
    node_name: str,
    decision_payload: dict,
) -> None:
    """Record execution order and per-node decisions in a structured format."""
    _ensure_metadata(state)
    state["metadata"]["agent_sequence"].append(node_name)
    state["metadata"]["decisions"][node_name] = decision_payload


class MathInquiriesGraph:
    """
    LangGraph workflow for Math Inquiries multi-agent system.

    Orchestrates: Planner → Retriever → Summarizer → Reflector → ToolAgent

    With feedback loop: Reflector can route back to Planner if confidence is low.
    """

    def __init__(self, retriever: ChromaDBRetriever = None):
        """
        Initialize the workflow graph.

        Args:
            retriever: ChromaDBRetriever instance for retrieval
        """
        self.retriever = retriever or ChromaDBRetriever()
        self.graph = self._build_graph()
        self.compiled_graph = self.graph.compile()

    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph StateGraph.

        Returns:
            StateGraph instance (not yet compiled)
        """
        # Initialize state graph
        graph = StateGraph(WorkflowState)

        # Instantiate agents
        planner = PlannerAgent()
        retriever = RetrieverAgent(self.retriever)
        summarizer = SummarizerAgent(use_llm=True)
        reflector = ReflectiveAgent(use_llm=not runtime_config.FAST_MODE)
        tool_agent = ToolAgent()

        # Define node functions (wrapper for agents)
        def planner_node(state: WorkflowState) -> WorkflowState:
            """Planner node: classify query and decide retrieval params."""
            started = time.perf_counter()
            updated = planner.run(state)
            elapsed = time.perf_counter() - started
            logger.info("Timing | planner | %.3fs", elapsed)
            updated["metadata"]["timestamps"]["planner"] = elapsed
            # Planner writes rich decision payload directly; keep sequence here.
            _record_node_metadata(
                updated,
                "planner",
                updated.get("metadata", {}).get("decisions", {}).get("planner", {}),
            )
            return updated

        def retriever_node(state: WorkflowState) -> WorkflowState:
            """Retriever node: RAG - get documents."""
            started = time.perf_counter()
            updated = retriever.run(state)
            elapsed = time.perf_counter() - started
            logger.info("Timing | retriever | %.3fs", elapsed)
            updated["metadata"]["timestamps"]["retriever_total"] = elapsed
            docs = updated.get("retrieved_docs", [])
            _record_node_metadata(
                updated,
                "retriever",
                {
                    "requested_k": updated.get("metadata", {})
                    .get("decisions", {})
                    .get("planner", {})
                    .get("k_documents"),
                    "retrieved_count": len(docs),
                    "top_sources": sorted(
                        {
                            doc.get("source", "unknown")
                            for doc in docs
                            if isinstance(doc, dict)
                        }
                    ),
                },
            )
            return updated

        def summarizer_node(state: WorkflowState) -> WorkflowState:
            """Summarizer node: generate summary."""
            started = time.perf_counter()
            updated = summarizer.run(state)
            elapsed = time.perf_counter() - started
            logger.info("Timing | summarizer | %.3fs", elapsed)
            updated["metadata"]["timestamps"]["summarizer_total"] = elapsed
            summary_text = updated.get("summary", "")
            _record_node_metadata(
                updated,
                "summarizer",
                {
                    "mode": (
                        "llm" if getattr(summarizer, "llm_ready", False) else "fallback"
                    ),
                    "summary_word_count": len(summary_text.split()),
                    "had_documents": len(updated.get("retrieved_docs", [])) > 0,
                },
            )
            return updated

        def reflector_node(state: WorkflowState) -> WorkflowState:
            """Reflector node: evaluate quality."""
            started = time.perf_counter()
            updated = reflector.run(state)
            elapsed = time.perf_counter() - started
            logger.info("Timing | reflector | %.3fs", elapsed)
            updated["metadata"]["timestamps"]["reflector_total"] = elapsed
            metrics = updated.get("reflection_metrics", {})
            _record_node_metadata(
                updated,
                "reflector",
                {
                    "confidence": metrics.get("confidence"),
                    "should_retry": metrics.get("should_retry"),
                    "evaluation_source": metrics.get("evaluation_source"),
                },
            )
            return updated

        def tool_node(state: WorkflowState) -> WorkflowState:
            """Tool node: execute external tools.

            Example usage:
                tool_agent = ToolAgent()
                state = tool_agent.run(state)
            """
            started = time.perf_counter()
            updated = tool_agent.run(state)
            elapsed = time.perf_counter() - started
            logger.info("Timing | tool_agent | %.3fs", elapsed)
            updated["metadata"]["timestamps"]["tool_agent_total"] = elapsed
            _record_node_metadata(
                updated,
                "tool_agent",
                updated.get("metadata", {}).get("decisions", {}).get("tool_agent", {}),
            )
            return updated

        # Add nodes
        graph.add_node("planner", planner_node)
        graph.add_node("retriever", retriever_node)
        graph.add_node("summarizer", summarizer_node)
        graph.add_node("reflector", reflector_node)
        graph.add_node("tool_agent", tool_node)

        # Add edges: define the flow
        graph.add_edge("planner", "retriever")
        graph.add_edge("retriever", "summarizer")
        graph.add_edge("summarizer", "reflector")

        # Conditional edge: reflector decides continue or retry via planner.
        graph.add_conditional_edges(
            "reflector",
            self._should_retry,
            {
                "retry": "planner",  # Go back to planner for plan adjustment
                "continue": "tool_agent",  # Go to tool agent
            },
        )

        # Final edge: tool agent -> END
        graph.add_edge("tool_agent", END)

        # Set entry point
        graph.set_entry_point("planner")

        logger.info("LangGraph workflow graph constructed")
        return graph

    def _should_retry(self, state: WorkflowState) -> Literal["retry", "continue"]:
        """
        Conditional routing: decide if we should re-retrieve or continue.

        Decision logic:
        - If confidence < threshold AND iterations < max: retry retrieval
        - Otherwise: continue to tool agent and finish

        Args:
            state: Current workflow state

        Returns:
            "retry" to go back to retriever, "continue" to proceed to tools
        """
        # If no metrics yet, continue to tool phase (safe default).
        if not state["reflection_metrics"]:
            return "continue"

        # FAST_MODE: skip iterative retry loop for near-instant responses.
        if runtime_config.FAST_MODE:
            state["metadata"].setdefault("trace", []).append(
                {
                    "node": "reflector",
                    "iteration": state["iteration_count"],
                    "decision": "continue",
                    "reason": "fast_mode",
                }
            )
            logger.info("FAST_MODE enabled: skipping retry loop")
            return "continue"

        # Check confidence
        confidence = state["reflection_metrics"].get("confidence", 0.0)
        iteration = state["iteration_count"]

        should_retry = (
            confidence < reflection_config.CONFIDENCE_THRESHOLD
            and iteration < reflection_config.MAX_ITERATIONS
        )

        if should_retry:
            logger.info(
                f"Confidence {confidence:.2f} < threshold {reflection_config.CONFIDENCE_THRESHOLD}. "
                f"Re-planning/re-retrieving (next iteration {iteration + 1})."
            )
            state["metadata"].setdefault("trace", []).append(
                {
                    "node": "reflector",
                    "iteration": iteration,
                    "confidence": confidence,
                    "decision": "retry",
                }
            )
            return "retry"
        else:
            if confidence >= reflection_config.CONFIDENCE_THRESHOLD:
                logger.info(
                    f"Confidence {confidence:.2f} >= threshold. Proceeding to tools."
                )
            else:
                logger.info(
                    f"Max iterations ({reflection_config.MAX_ITERATIONS}) reached. "
                    "Stopping re-retrieval despite low confidence."
                )
            state["metadata"].setdefault("trace", []).append(
                {
                    "node": "reflector",
                    "iteration": iteration,
                    "confidence": confidence,
                    "decision": "continue",
                }
            )
            return "continue"

    def run(self, user_query: str) -> dict:
        """
        Execute the workflow for a user query.

        Args:
            user_query: User's question

        Returns:
            Final state dict with all results
        """
        logger.info(f"Starting workflow for query: {user_query}")
        workflow_started = time.perf_counter()

        # Initialize state
        initial_state = initialize_state(user_query)

        # Execute graph
        final_state = self.compiled_graph.invoke(initial_state)

        total_elapsed = time.perf_counter() - workflow_started
        final_state["metadata"]["timestamps"]["workflow_total"] = total_elapsed

        logger.info(
            f"Workflow completed after {final_state['iteration_count']} iterations"
        )
        logger.info("Timing | workflow_total | %.3fs", total_elapsed)
        return final_state

    def get_graph_structure(self) -> str:
        """
        Return a text representation of the graph structure for debugging.

        Returns:
            String describing the graph
        """
        return self.graph.get_graph().draw_mermaid()

    @staticmethod
    def format_execution_trace(final_state: dict) -> str:
        """Return a readable trace string from workflow metadata."""
        trace = final_state.get("metadata", {}).get("trace", [])
        if not trace:
            return "(no trace available)"

        lines = ["Execution trace:"]
        for event in trace:
            node = event.get("node", "unknown")
            iteration = event.get("iteration", "?")
            confidence = event.get("confidence")
            decision = event.get("decision", event.get("next_action", ""))
            k_docs = event.get("k_documents")

            parts = [f"- {node} iter={iteration}"]
            if k_docs is not None:
                parts.append(f"k={k_docs}")
            if confidence is not None:
                parts.append(f"confidence={confidence:.2f}")
            if decision:
                parts.append(f"decision={decision}")
            lines.append(" | ".join(parts))
        return "\n".join(lines)


def create_graph(retriever: ChromaDBRetriever = None) -> MathInquiriesGraph:
    """
    Factory function to create and return the workflow graph.

    Args:
        retriever: ChromaDBRetriever instance (optional)

    Returns:
        MathInquiriesGraph instance
    """
    return MathInquiriesGraph(retriever)
