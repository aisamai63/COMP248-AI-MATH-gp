"""
Utility functions for Math Inquiries prototype.
"""

from typing import Any


def safe_truncate(text: str, n: int = 300) -> str:
    """
    Truncate text to n characters, adding ellipsis if needed.
    Args:
        text: Input string.
        n: Max length.
    Returns:
        Truncated string.
    """
    return text if len(text) <= n else text[:n] + "..."


# Add more utility functions as needed (e.g., text cleaning, scoring)
