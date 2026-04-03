"""Registry compatibility module.

Provides canonical package-qualified import target:
    from prototype.tools.registry import ToolRegistry, ToolSpec
"""

from prototype.tools.tool_registry import ToolRegistry, ToolSpec

__all__ = ["ToolRegistry", "ToolSpec"]
