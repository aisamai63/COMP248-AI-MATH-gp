"""Built-in tool functions: Sympy calculator and web search."""

import json
import importlib
from urllib.parse import quote_plus
from urllib.request import urlopen
from typing import List

from prototype.tools.registry import ToolRegistry, ToolSpec
from prototype.config import tool_config


def _contains_any(query: str, keywords: List[str]) -> bool:
    return any(kw in query for kw in keywords)


def sympy_calculator_tool(query: str) -> str:
    """Evaluate/simplify/solve mathematical expressions with Sympy."""
    try:
        import sympy

        q = query.lower().strip()

        # Basic cleanup for natural language prefixes.
        for prefix in [
            "calculate",
            "compute",
            "evaluate",
            "solve",
            "simplify",
            "factor",
        ]:
            if q.startswith(prefix):
                q = q[len(prefix) :].strip(" :")

        if "=" in q:
            left, right = q.split("=", 1)
            x = sympy.symbols("x")
            equation = sympy.Eq(
                sympy.sympify(left.strip()), sympy.sympify(right.strip() or "0")
            )
            solutions = sympy.solve(equation, x)
            return f"Solutions: {solutions}"

        expression = sympy.sympify(q)
        simplified = sympy.simplify(expression)
        return f"Result: {simplified}"
    except Exception as exc:
        return f"Calculator failed: {exc}"


def serpapi_web_search_tool(query: str) -> str:
    """Web search via SerpAPI when SERPAPI_KEY is configured."""
    api_key = tool_config.SERPAPI_KEY.strip()
    if not api_key:
        return "SerpAPI key is not configured (SERPAPI_KEY)."

    try:
        serpapi_module = importlib.import_module("serpapi")
        GoogleSearch = getattr(serpapi_module, "GoogleSearch")

        params = {
            "q": query,
            "api_key": api_key,
            "num": tool_config.WEB_SEARCH_RESULTS_COUNT,
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        organic = results.get("organic_results", [])[:3]
        if not organic:
            return "No web results found via SerpAPI."

        lines = ["SerpAPI results:"]
        for idx, item in enumerate(organic, 1):
            title = item.get("title", "No title")
            snippet = item.get("snippet", "No snippet")
            lines.append(f"{idx}. {title}")
            lines.append(f"   {snippet}")
        return "\n".join(lines)
    except Exception as exc:
        return f"SerpAPI search failed: {exc}"


def duckduckgo_web_search_tool(query: str) -> str:
    """Web search fallback using DuckDuckGo Instant Answer API (no extra dependency)."""
    try:
        url = (
            "https://api.duckduckgo.com/?q="
            + quote_plus(query)
            + "&format=json&no_html=1&no_redirect=1"
        )
        with urlopen(url, timeout=tool_config.DUCKDUCKGO_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))

        abstract = payload.get("AbstractText", "").strip()
        related = payload.get("RelatedTopics", [])

        if abstract:
            return f"DuckDuckGo answer: {abstract}"

        lines = ["DuckDuckGo related results:"]
        count = 0
        for item in related:
            # Some entries are nested.
            if isinstance(item, dict) and "Text" in item:
                lines.append(f"- {item['Text']}")
                count += 1
            elif isinstance(item, dict) and "Topics" in item:
                for sub in item.get("Topics", []):
                    if isinstance(sub, dict) and "Text" in sub:
                        lines.append(f"- {sub['Text']}")
                        count += 1
                    if count >= 3:
                        break
            if count >= 3:
                break

        if count == 0:
            return "DuckDuckGo returned no useful results."
        return "\n".join(lines)
    except Exception as exc:
        return f"DuckDuckGo search failed: {exc}"


def web_search_tool(query: str) -> str:
    """Use SerpAPI if configured; otherwise fallback to DuckDuckGo."""
    serp_res = serpapi_web_search_tool(query)
    if not serp_res.startswith(
        "SerpAPI key is not configured"
    ) and not serp_res.startswith("SerpAPI search failed"):
        return serp_res
    return duckduckgo_web_search_tool(query)


def register_default_tools(registry: ToolRegistry) -> None:
    """Register built-in calculator and web-search tools."""
    registry.add_tool(
        ToolSpec(
            name="calculator",
            func=sympy_calculator_tool,
            description="Symbolic calculator using Sympy",
            selector=lambda q: _contains_any(
                q,
                [
                    "calculate",
                    "compute",
                    "solve",
                    "evaluate",
                    "simplify",
                    "factor",
                    "derivative",
                    "integral",
                    "equation",
                ],
            ),
        )
    )

    registry.add_tool(
        ToolSpec(
            name="web_search",
            func=web_search_tool,
            description="Web search via SerpAPI (or DuckDuckGo fallback)",
            selector=lambda q: _contains_any(
                q,
                [
                    "latest",
                    "current",
                    "recent",
                    "news",
                    "today",
                    "update",
                    "2025",
                    "2026",
                ],
            ),
        )
    )
