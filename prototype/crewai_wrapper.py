"""Minimal CrewAI wrapper around the existing LangGraph workflow.

This keeps LangGraph as the reasoning source of truth while satisfying the
prototype requirement to demonstrate a real CrewAI runtime path.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from crewai import Agent, Crew, LLM, Process, Task
from crewai.tools import tool

from prototype.config import llm_config
from prototype.workflow.graph import MathInquiriesGraph, create_graph

logger = logging.getLogger(__name__)


def _build_crewai_llm() -> Optional[LLM]:
    """Build a CrewAI LLM configured from the project environment."""
    provider = llm_config.LLM_PROVIDER

    if provider == "openai":
        if not llm_config.OPENAI_API_KEY:
            return None
        return LLM(
            model=llm_config.OPENAI_MODEL,
            api_key=llm_config.OPENAI_API_KEY,
            provider="openai",
            temperature=0.0,
        )

    if provider == "gemini":
        if not llm_config.GEMINI_API_KEY:
            return None
        return LLM(
            model=llm_config.GEMINI_MODEL,
            api_key=llm_config.GEMINI_API_KEY,
            provider="google",
            temperature=0.0,
        )

    if not llm_config.MISTRAL_API_KEY:
        return None
    return LLM(
        model=llm_config.MISTRAL_MODEL,
        api_key=llm_config.MISTRAL_API_KEY,
        provider="mistral",
        temperature=0.0,
    )


class CrewAILangGraphWorkflow:
    """Single-agent CrewAI wrapper that delegates work to LangGraph."""

    def __init__(self, langgraph_workflow: MathInquiriesGraph | None = None):
        self.langgraph_workflow = langgraph_workflow or create_graph()
        self.crew_llm = _build_crewai_llm()
        self.last_langgraph_state: Dict[str, Any] | None = None
        self.tool = self._build_langgraph_tool()
        self.crew_enabled = self.crew_llm is not None
        self.agent: Agent | None = None
        self.task: Task | None = None
        self.crew: Crew | None = None

        if not self.crew_enabled:
            logger.warning(
                "CrewAI LLM configuration is missing; the wrapper will fall back "
                "to direct LangGraph execution."
            )
            return

        self.agent = Agent(
            role="Math workflow orchestrator",
            goal="Use the LangGraph tool once and return its grounded output.",
            backstory=(
                "You are a thin CrewAI orchestration layer over an existing "
                "LangGraph math system. Your only job is to call the tool on "
                "the user's query and preserve the resulting state."
            ),
            verbose=False,
            allow_delegation=False,
            tools=[self.tool],
            llm=self.crew_llm,
            function_calling_llm=self.crew_llm,
            max_iter=3,
            reasoning=False,
        )
        self.task = Task(
            description=(
                "Call the run_langgraph_workflow tool exactly once with the "
                "user query below. Do not invent facts.\n\n"
                "User query: {query}"
            ),
            expected_output=(
                "A faithful answer derived from the LangGraph workflow state, "
                "including any relevant tool or reflection information."
            ),
            agent=self.agent,
            markdown=True,
        )
        self.crew = Crew(
            agents=[self.agent],
            tasks=[self.task],
            process=Process.sequential,
            verbose=False,
            function_calling_llm=self.crew_llm,
        )

    def _build_langgraph_tool(self):
        """Create a CrewAI tool that invokes the underlying LangGraph workflow."""

        @tool("run_langgraph_workflow", result_as_answer=True)
        def run_langgraph_workflow(query: str) -> str:
            """Run the LangGraph workflow for a single user query."""
            state = self.langgraph_workflow.run(query)
            self.last_langgraph_state = state
            return json.dumps(state, ensure_ascii=False, default=str)

        return run_langgraph_workflow

    def run(self, user_query: str) -> Dict[str, Any]:
        """Run the CrewAI wrapper and return the underlying LangGraph state."""
        self.last_langgraph_state = None

        if not self.crew_enabled or self.crew is None:
            state = self.langgraph_workflow.run(user_query)
            state.setdefault("metadata", {}).setdefault("crew", {})
            state["metadata"]["crew"].update(
                {
                    "enabled": False,
                    "used": False,
                    "mode": "direct_langgraph_fallback",
                }
            )
            return state

        try:
            crew_result = self.crew.kickoff(inputs={"query": user_query})
        except Exception as exc:
            logger.warning("CrewAI kickoff failed; falling back to LangGraph: %s", exc)
            state = self.langgraph_workflow.run(user_query)
            state.setdefault("metadata", {}).setdefault("crew", {})
            state["metadata"]["crew"].update(
                {
                    "enabled": True,
                    "used": False,
                    "mode": "fallback_after_crew_error",
                    "error": str(exc),
                }
            )
            return state

        state = self.last_langgraph_state or self.langgraph_workflow.run(user_query)
        state.setdefault("metadata", {}).setdefault("crew", {})
        state["metadata"]["crew"].update(
            {
                "enabled": True,
                "used": self.last_langgraph_state is not None,
                "mode": "crewai_sequential_wrapper",
                "raw_output": getattr(crew_result, "raw", None) or str(crew_result),
            }
        )
        return state

    def get_graph_structure(self) -> str:
        """Proxy the underlying LangGraph structure for debugging."""
        return self.langgraph_workflow.get_graph_structure()

    @staticmethod
    def format_execution_trace(final_state: dict) -> str:
        """Proxy the LangGraph trace formatter."""
        return MathInquiriesGraph.format_execution_trace(final_state)


def create_crewai_workflow(
    langgraph_workflow: MathInquiriesGraph | None = None,
) -> CrewAILangGraphWorkflow:
    """Factory for the CrewAI-backed workflow wrapper."""
    return CrewAILangGraphWorkflow(langgraph_workflow)
