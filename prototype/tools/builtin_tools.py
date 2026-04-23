"""Built-in tool functions: Sympy calculator and web search."""

import ast
import json
import importlib
import re
import unicodedata
from urllib.parse import quote_plus
from urllib.request import urlopen
from typing import List

from prototype.tools.registry import ToolRegistry, ToolSpec
from prototype.config import tool_config


def _contains_any(query: str, keywords: List[str]) -> bool:
    return any(kw in query for kw in keywords)


def _looks_like_math_query(query: str) -> bool:
    """Heuristic: detect expression/equation/matrix-like math input."""
    q = _normalize_math_text(query.lower())
    if any(
        token in q
        for token in [
            "calculate",
            "compute",
            "solve",
            "evaluate",
            "simplify",
            "factor",
            "derivative",
            "integral",
            "equation",
            "matrix",
            "row echelon",
            "rref",
            "ref",
        ]
    ):
        return True

    if "=" in q and re.search(r"[a-z]", q):
        return True

    # Typical symbolic math expression signal.
    if re.search(r"[0-9][xya-z]|[a-z][0-9]|[+\-*/^()]", q):
        return bool(re.search(r"[0-9]|[a-z]", q))

    return False


def _format_matrix_block(matrix) -> str:
    """Render a matrix in a readable multiline ASCII block."""
    rows = []
    for row_idx in range(matrix.rows):
        entries = [str(matrix[row_idx, col_idx]) for col_idx in range(matrix.cols)]
        rows.append("[ " + "  ".join(entries) + " ]")
    return "\n".join(rows)


def _format_sympy_latex(expr) -> str:
    """Best-effort LaTeX formatting for Sympy objects."""
    try:
        import sympy

        latex_str = sympy.latex(expr)
        # Sympy matrices default to \\left[\\begin{matrix} ... \\end{matrix}\\right]
        # which renders fine, but bmatrix tends to look nicer in math UIs.
        if "\\begin{matrix}" in latex_str:
            latex_str = latex_str.replace("\\left[\\begin{matrix}", "\\begin{bmatrix}")
            latex_str = latex_str.replace("\\end{matrix}\\right]", "\\end{bmatrix}")
        return latex_str
    except Exception:
        return ""


def _wrap_math_block(latex_str: str) -> str:
    latex_str = (latex_str or "").strip()
    if not latex_str:
        return ""
    return f"$$\n{latex_str}\n$$"


def _sanitize_equation_side(side: str) -> str:
    """Keep only the leading math-like segment of one equation side."""
    allowed_funcs = {"sqrt", "sin", "cos", "tan", "log", "ln", "exp"}
    tokens = side.strip().split()
    kept = []

    for token in tokens:
        lower = token.lower()
        # Stop at obvious natural-language words (e.g., "convert", "this").
        if lower.isalpha() and len(lower) > 1 and lower not in allowed_funcs:
            break
        # Keep math-ish tokens.
        if re.fullmatch(r"[0-9a-zA-Z()+\-*/.^]+", token):
            kept.append(token)
        else:
            # Unknown token: stop before accidental prose/noise.
            break

    if not kept:
        return ""
    return " ".join(kept)


def _normalize_math_text(text: str) -> str:
    """Normalize common Unicode math symbols to ASCII-friendly text."""
    replacements = {
        "\u2212": "-",  # minus sign
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u00d7": "*",  # multiplication sign
        "\u22c5": "*",  # dot operator
        "\u00b7": "*",  # middle dot
        "\u00f7": "/",  # division sign
        "\u2264": "<=",
        "\u2265": ">=",
        "\u2260": "!=",
        "\u221a": "sqrt",
    }
    out = text
    for old, new in replacements.items():
        out = out.replace(old, new)

    # Remove zero-width/BOM and other non-printable control/format characters.
    out = out.replace("\ufeff", "")
    out = out.replace("\u200b", "")
    out = out.replace("\u200c", "")
    out = out.replace("\u200d", "")
    out = out.replace("\u2060", "")
    out = "".join(
        ch
        for ch in out
        if (ch in "\n\t " or not unicodedata.category(ch).startswith("C"))
    )

    out = out.replace("^", "**")
    # Merge variable subscripts that often appear spaced when copied from PDFs:
    # "x 1" -> "x1", "a 12" -> "a12".
    out = re.sub(r"\b([a-zA-Z])\s+(\d+)\b", r"\1\2", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


def _extract_equation_candidates(text: str) -> List[str]:
    """Extract equation-like substrings from free-form text."""
    normalized = _normalize_math_text(text)

    # Remove common command prefixes only at start.
    normalized = re.sub(
        r"^(calculate|compute|evaluate|solve|simplify|factor|solve this)\b[:\s-]*",
        "",
        normalized,
        flags=re.IGNORECASE,
    )

    # Insert line breaks before likely equation starts if query is one long line.
    normalized = re.sub(
        r"\s(?=[+\-]?\d*[a-zA-Z(][^=]{0,30}=)",
        "\n",
        normalized,
    )

    pieces = []
    for raw_piece in re.split(r"[\n;]+", normalized):
        piece = raw_piece.strip(" ,")
        if "=" not in piece:
            continue

        # If a sentence still contains multiple equations, split on equation boundaries.
        matches = re.findall(
            r"([+\-]?\s*[0-9a-zA-Z()*/.+\-\s]+?=[+\-]?\s*[0-9a-zA-Z()*/.+\-\s]+?)(?=(?:\s+[+\-]?\d*[a-zA-Z(][^=]{0,30}=)|$)",
            piece,
        )
        if matches:
            pieces.extend(m.strip() for m in matches)
        else:
            pieces.append(piece)

    cleaned = []
    for candidate in pieces:
        # Trim non-math prefix (e.g., "this 2x+y=3" -> "2x+y=3").
        start_match = re.search(r"[+\-]?\d*[a-zA-Z(]", candidate)
        if start_match:
            candidate = candidate[start_match.start() :]
        if "=" in candidate:
            cleaned.append(candidate.strip())

    return cleaned


def sympy_calculator_tool(query: str) -> str:
    """Evaluate/simplify/solve mathematical expressions with Sympy."""
    try:
        import sympy
        from sympy.parsing.sympy_parser import (
            standard_transformations,
            implicit_multiplication_application,
            parse_expr,
        )

        q = _normalize_math_text(query.lower().strip())
        transformations = standard_transformations + (
            implicit_multiplication_application,
        )

        def _extract_balanced_matrix_literals(text: str, limit: int = 2) -> list[str]:
            """Extract up to `limit` balanced `[[...]]` literals from text."""
            literals: list[str] = []
            i = 0
            n = len(text)
            while i < n and len(literals) < limit:
                start = text.find("[[", i)
                if start == -1:
                    break
                depth = 0
                j = start
                while j < n:
                    if text[j] == "[":
                        depth += 1
                    elif text[j] == "]":
                        depth -= 1
                        if depth == 0:
                            literals.append(text[start : j + 1])
                            i = j + 1
                            break
                    j += 1
                else:
                    break
            return literals

        # Matrix operations: handle common "A=[[...]] and B=[[...]]" inputs.
        # This bypasses parse_expr quirks on nested list literals.
        if "[[" in q and "]]" in q:
            try:
                mats = _extract_balanced_matrix_literals(q, limit=2)
                matrices = [sympy.Matrix(ast.literal_eval(m)) for m in mats]
                if matrices:
                    a = matrices[0]
                    b = matrices[1] if len(matrices) > 1 else None

                    if ("inverse" in q or " inv" in q) and b is None:
                        inv_a = a.inv()
                        latex = _wrap_math_block(_format_sympy_latex(inv_a))
                        return "Inverse:\n" + str(inv_a) + ("\n\n" + latex if latex else "")

                    if "det" in q or "determinant" in q:
                        det_a = a.det()
                        latex = _wrap_math_block(_format_sympy_latex(det_a))
                        return f"Determinant: {det_a}" + ("\n\n" + latex if latex else "")

                    if b is not None and any(tok in q for tok in ["*", " x ", "×", "multiply"]):
                        prod = a * b
                        latex = _wrap_math_block(_format_sympy_latex(prod))
                        return "Product A·B:\n" + str(prod) + ("\n\n" + latex if latex else "")
            except Exception:
                # Fall through to generic parsing.
                pass

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

        equation_texts = _extract_equation_candidates(q)
        if equation_texts:
            equations = []
            all_symbols = set()

            for eq_text in equation_texts:
                left, right = eq_text.split("=", 1)
                left_clean = _sanitize_equation_side(left)
                right_clean = _sanitize_equation_side(right)
                if not left_clean or not right_clean:
                    continue
                left_expr = parse_expr(
                    left_clean,
                    transformations=transformations,
                    evaluate=True,
                )
                right_expr = parse_expr(
                    right_clean,
                    transformations=transformations,
                    evaluate=True,
                )
                equation = sympy.Eq(left_expr, right_expr)
                equations.append(equation)
                all_symbols.update(equation.free_symbols)

            if not equations:
                return "Could not parse valid equation(s). Please use format like: 2x+y-z=3; 4x-y+2z=1; -2x+3y+z=4"

            if not all_symbols:
                return "No variables found in the provided equation(s)."

            symbols = sorted(all_symbols, key=lambda s: s.name)

            # If this is a linear system, return structured matrix workflow (A|b, REF, RREF, solution).
            if len(equations) >= 2:
                try:
                    a_matrix, b_matrix = sympy.linear_eq_to_matrix(equations, symbols)
                    aug_matrix = a_matrix.row_join(b_matrix)
                    ref_matrix = aug_matrix.echelon_form()
                    rref_matrix, _ = aug_matrix.rref()
                    solutions = sympy.solve(equations, symbols, dict=True)

                    lines = [
                        "Augmented matrix [A|b]:",
                        _format_matrix_block(aug_matrix),
                        "",
                        "Row echelon form (REF):",
                        _format_matrix_block(ref_matrix),
                        "",
                        "Reduced row echelon form (RREF):",
                        _format_matrix_block(rref_matrix),
                    ]

                    if not solutions:
                        lines.extend(
                            ["", "Solution: No solution (inconsistent system)."]
                        )
                    elif len(solutions) == 1:
                        sol = solutions[0]
                        sol_parts = [
                            f"{sym} = {sympy.simplify(sol[sym])}"
                            for sym in symbols
                            if sym in sol
                        ]
                        lines.extend(["", "Solution: " + ", ".join(sol_parts)])
                    else:
                        lines.extend(["", f"Solutions: {solutions}"])

                    # Add LaTeX blocks to improve visual rendering (Streamlit will render $$...$$).
                    latex_chunks = []
                    for title, mat in [
                        ("Augmented matrix [A|b]", aug_matrix),
                        ("Row echelon form (REF)", ref_matrix),
                        ("Reduced row echelon form (RREF)", rref_matrix),
                    ]:
                        latex = _format_sympy_latex(mat)
                        if latex:
                            latex_chunks.append(f"{title}:\n{_wrap_math_block(latex)}")
                    if latex_chunks:
                        lines.extend(["", "LaTeX view:", *latex_chunks])

                    return "\n".join(lines)
                except Exception:
                    # Fall back to generic equation solving path if not linear.
                    pass

            solutions = sympy.solve(equations, symbols, dict=True)

            if not solutions:
                return "No solution found (system may be inconsistent)."

            if len(solutions) == 1:
                solution = solutions[0]
                parts = [
                    f"{sym} = {sympy.simplify(solution[sym])}"
                    for sym in symbols
                    if sym in solution
                ]
                eqs = []
                for sym in symbols:
                    if sym not in solution:
                        continue
                    rhs = sympy.simplify(solution[sym])
                    eqs.append(sympy.Eq(sym, rhs))
                latex_lines = [sympy.latex(e) for e in eqs] if eqs else []
                latex_block = _wrap_math_block("\\\\\n".join(latex_lines)) if latex_lines else ""
                pretty = "Solution: " + ", ".join(parts)
                return pretty + ("\n\n" + latex_block if latex_block else "")

            return f"Solutions: {solutions}"

        if not _looks_like_math_query(q):
            return "Calculator expects a math expression or equation system. Example: 2x+y-z=3; 4x-y+2z=1; -2x+3y+z=4"

        expression = parse_expr(q, transformations=transformations, evaluate=True)
        simplified = sympy.simplify(expression)
        latex = _format_sympy_latex(simplified)
        latex_block = _wrap_math_block(latex) if latex else ""
        pretty = f"Result: {simplified}"
        return pretty + ("\n\n" + latex_block if latex_block else "")
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
            selector=lambda q: _looks_like_math_query(q),
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
