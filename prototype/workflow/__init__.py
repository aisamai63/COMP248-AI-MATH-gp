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

from prototype.workflow.state import WorkflowState, initialize_state, ReflectionMetrics
from prototype.workflow.graph import create_graph, MathInquiriesGraph

__all__ = [
    "WorkflowState",
    "initialize_state",
    "ReflectionMetrics",
    "MathInquiriesGraph",
    "create_graph",
]
