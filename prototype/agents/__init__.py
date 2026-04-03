"""
Agents package: individual agent implementations.

Agents are modular components of the multi-agent system.
Each agent has a specific role and is coordinated by LangGraph.

Agents:
- PlannerAgent: Routes and makes high-level decisions
- RetrieverAgent: RAG - retrieves relevant documents
- SummarizerAgent: Generates LLM-based summaries
- ReflectiveAgent: Evaluates summary quality
- ToolAgent: Executes external tools/APIs
"""

from prototype.agents.base import BaseAgent
from prototype.agents.planner import PlannerAgent
from prototype.agents.retriever import RetrieverAgent
from prototype.agents.summarizer import SummarizerAgent
from prototype.agents.reflective_agent import ReflectiveAgent
from prototype.agents.tool_agent import ToolAgent

__all__ = [
    "BaseAgent",
    "PlannerAgent",
    "RetrieverAgent",
    "SummarizerAgent",
    "ReflectiveAgent",
    "ToolAgent",
]
