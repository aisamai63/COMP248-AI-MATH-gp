"""
Workflow state management for LangGraph-based multi-agent system.
Defines the data structure that flows through each agent node.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from typing_extensions import TypedDict


class WorkflowState(TypedDict):
    """
    State object passed between agents in the LangGraph workflow.

    Data Flow:
    - Planner receives: user_query
    - SearchAgent receives: user_query; returns: retrieved_docs
    - SummarizerAgent receives: retrieved_docs, user_query; returns: summary
    - ReflectiveAgent receives: summary, retrieved_docs, user_query; returns: reflection_metrics
    - ToolAgent (optional) receives: user_query, summary; returns: tool_results

    Attributes:
        user_query: Original user query (never modified)
        retrieved_docs: List of documents returned by search
        summary: Generated summary of retrieved documents
        reflection_metrics: Evaluation scores (confidence, factuality, etc.)
        tool_results: Results from any external tools used
        iteration_count: Number of retrieval iterations so far
        should_continue: Whether workflow should continue or end
        error_message: Error message if something failed
        metadata: Additional tracking info (timestamps, agent decisions, etc.)
    """

    # Core flow data
    user_query: str
    retrieved_docs: List[Dict[str, Any]]
    summary: str

    # Evaluation data
    reflection_metrics: Dict[str, Any]

    # Tool data
    tool_results: Optional[str]

    # Control flow
    iteration_count: int
    should_continue: bool

    # Error handling
    error_message: Optional[str]

    # Metadata for tracking
    metadata: Dict[str, Any]


@dataclass
class ReflectionMetrics:
    """
    Structured reflection metrics returned by ReflectiveAgent.

    Each metric is on a 0-1 scale where 1 is ideal.
    """

    coverage: float  # Does summary address the query? (0-1)
    factuality: float  # Are claims supported by sources? (0-1)
    conciseness: float  # Is summary appropriately brief? (0-1)
    coherence: float  # Is summary well-structured? (0-1)

    # Derived metrics
    confidence: float  # Composite score (weighted average of above)

    # Feedback
    notes: List[str] = field(default_factory=list)  # Actionable notes
    should_retry: bool = False  # Should Planner re-retrieve?


def initialize_state(user_query: str) -> WorkflowState:
    """
    Initialize a fresh workflow state for a new query.

    Args:
        user_query: User's question or query

    Returns:
        Fresh WorkflowState with defaults
    """
    return WorkflowState(
        user_query=user_query,
        retrieved_docs=[],
        summary="",
        reflection_metrics={},
        tool_results=None,
        iteration_count=0,
        should_continue=True,
        error_message=None,
        metadata={
            "agent_sequence": [],  # Record which agents ran
            "timestamps": {},  # When each agent ran
            "decisions": {},  # Why agent made decision
            "llm_calls": [],  # Per-agent LLM call diagnostics
        },
    )


def log_agent_execution(
    state: WorkflowState,
    agent_name: str,
    execution_time: float,
    decision: Optional[str] = None,
) -> None:
    """
    Log an agent's execution in the state metadata.
    Useful for debugging and understanding the workflow.

    Args:
        state: Current workflow state (modified in-place)
        agent_name: Name of agent that just executed
        execution_time: How long the agent took (seconds)
        decision: Optional decision made by the agent
    """
    state["metadata"]["agent_sequence"].append(agent_name)
    state["metadata"]["timestamps"][agent_name] = execution_time
    if decision:
        state["metadata"]["decisions"][agent_name] = decision
