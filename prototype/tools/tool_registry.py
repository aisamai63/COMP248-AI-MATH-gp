"""Tool registry system for callable tools with pluggable selection rules."""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Any


@dataclass
class ToolSpec:
    """Definition of a callable tool and its selection logic."""

    name: str
    func: Callable[[str], str]
    description: str
    selector: Optional[Callable[[str], bool]] = None


class ToolRegistry:
    """Registry for adding, listing, selecting, and executing tools."""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    def add_tool(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def has_tool(self, tool_name: str) -> bool:
        return tool_name in self._tools

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())

    def get_tool(self, tool_name: str) -> Optional[ToolSpec]:
        return self._tools.get(tool_name)

    def select_by_query(self, query: str) -> List[str]:
        selected: List[str] = []
        q = query.strip().lower()
        for name, spec in self._tools.items():
            if spec.selector and spec.selector(q):
                selected.append(name)
        return selected

    def execute(self, tool_name: str, query: str) -> str:
        spec = self.get_tool(tool_name)
        if spec is None:
            return f"Tool not found: {tool_name}"
        try:
            return spec.func(query)
        except Exception as exc:
            return f"Tool execution failed ({tool_name}): {exc}"

    def execute_many(self, tool_names: List[str], query: str) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for name in tool_names:
            results.append({"tool": name, "result": self.execute(name, query)})
        return results
