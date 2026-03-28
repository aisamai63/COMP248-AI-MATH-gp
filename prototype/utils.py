def safe_truncate(text: str, n: int = 300) -> str:
    return text if len(text) <= n else text[:n] + "..."
