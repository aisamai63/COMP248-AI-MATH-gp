"""
Workflow package: LangGraph-based multi-agent orchestration.

The workflow coordinates execution of multiple agents using LangGraph,
a graph-based framework for building complex agent systems.

Key Components:
- state.py: Workflow state management and data structures
- graph.py: LangGraph workflow definition and orchestration
- config.py: Centralized configuration (see ../config.py)

Usage:
    from workflow import create_graph

    graph = create_graph()
    result = graph.run("What is the quadratic formula?")

    print(result["summary"])
    print(result["reflection_metrics"]["confidence"])
"""

import logging

from prototype.config import runtime_config
from prototype.workflow.state import WorkflowState, initialize_state, ReflectionMetrics
from prototype.workflow.graph import create_graph, MathInquiriesGraph

logger = logging.getLogger(__name__)


def create_runtime_graph(retriever=None):
    """Create the runtime workflow, optionally wrapped in CrewAI."""
    langgraph_workflow = create_graph(retriever)

    if not runtime_config.USE_CREWAI:
        return langgraph_workflow

    try:
        from prototype.crewai_wrapper import CrewAILangGraphWorkflow

        return CrewAILangGraphWorkflow(langgraph_workflow)
    except Exception as exc:
        logger.warning("CrewAI wrapper unavailable; using LangGraph directly: %s", exc)
        return langgraph_workflow


__all__ = [
    "WorkflowState",
    "initialize_state",
    "ReflectionMetrics",
    "MathInquiriesGraph",
    "create_graph",
    "create_runtime_graph",
]
