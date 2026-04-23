"""ToolAgent with registry-based callable tools and selection logic."""

import time
from typing import List

from prototype.agents.base import BaseAgent
from prototype.workflow.state import WorkflowState
from prototype.tools import ToolRegistry, register_default_tools


class ToolAgent(BaseAgent):
    """Executes selected tools based on planner hint and query intent."""

    name = "ToolAgent"

    def __init__(
        self,
        registry: ToolRegistry = None,
        *,
        allowed_tools: List[str] | None = None,
        decision_key: str = "tool_agent",
        expose_results: bool = True,
    ):
        super().__init__()
        self.registry = registry or ToolRegistry()
        if not self.registry.list_tools():
            register_default_tools(self.registry)
        self.allowed_tools = allowed_tools
        self.decision_key = decision_key
        self.expose_results = expose_results

    def run(self, state: WorkflowState) -> WorkflowState:
        start_time = time.time()
        try:
            query = state.get("user_query", "")
            selected_tools = self._select_tools(state, query)
            if self.allowed_tools is not None:
                selected_tools = [t for t in selected_tools if t in set(self.allowed_tools)]

            if not selected_tools:
                self._log_step("No tools selected", {"query": query})
                if self.expose_results:
                    state["tool_results"] = None
                outputs = []
            else:
                self._log_step("Selected tools", {"tools": selected_tools})
                outputs = self.registry.execute_many(selected_tools, query)
                if self.expose_results:
                    state["tool_results"] = self._format_tool_results(outputs)

            calculator_result = None
            for item in outputs:
                if item.get("tool") != "calculator":
                    continue
                result_text = str(item.get("result", "") or "")
                if result_text.strip() and not result_text.startswith("Calculator failed"):
                    calculator_result = result_text.strip()
                break

            elapsed = time.time() - start_time
            state["metadata"]["timestamps"][self.decision_key] = elapsed
            state["metadata"]["decisions"][self.decision_key] = {
                "selected_tools": selected_tools,
                "available_tools": self.registry.list_tools(),
                "outputs": outputs,
                "calculator_result": calculator_result,
            }
            return state
        except Exception as exc:
            return self._handle_error(state, exc)

    def _select_tools(self, state: WorkflowState, query: str) -> List[str]:
        """Select tools using planner instruction first, then query-based selection."""
        planner_decision = (
            state.get("metadata", {}).get("decisions", {}).get("planner", {})
        )
        requested_tools = planner_decision.get("tools", [])

        selected: List[str] = []

        # 1) Planner explicit tool instruction has priority.
        for tool_name in requested_tools:
            if self.registry.has_tool(tool_name) and tool_name not in selected:
                selected.append(tool_name)

        # 2) Query-driven inference as fallback/augment.
        inferred = self.registry.select_by_query(query)
        for tool_name in inferred:
            if tool_name not in selected:
                selected.append(tool_name)

        return selected

    def _format_tool_results(self, outputs: List[dict]) -> str:
        lines = ["## Tool Results"]
        for item in outputs:
            lines.append(f"\n### {item['tool']}")
            lines.append(str(item["result"]))
        return "\n".join(lines)


# Example usage in workflow node:
#   tool_agent = ToolAgent()
#   def tool_node(state: WorkflowState) -> WorkflowState:
#       return tool_agent.run(state)
