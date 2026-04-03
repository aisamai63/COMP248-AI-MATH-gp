"""
Configuration module for Math Inquiries prototype.
Centralizes all settings, constants, and environment variables.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class RAGConfig:
    """RAG (Retrieval-Augmented Generation) configuration."""

    # Retrieval settings
    K_DEFAULT: int = int(os.getenv("K_DEFAULT", "5"))
    K_EXPANDED: int = int(os.getenv("K_EXPANDED", "10"))
    K_MAX: int = int(os.getenv("K_MAX", "20"))

    # Embedding settings
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))

    # ChromaDB settings
    CHROMADB_COLLECTION_NAME: str = os.getenv("CHROMADB_COLLECTION_NAME", "math_docs")
    CHROMADB_DISTANCE_METRIC: str = os.getenv("CHROMADB_DISTANCE_METRIC", "cosine")


@dataclass
class LLMConfig:
    """LLM (Language Model) configuration."""

    # Provider selection: mistral | openai | gemini
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "mistral").strip().lower()

    # Mistral settings
    MISTRAL_API_KEY: str = os.getenv("MISTRAL_API_KEY", "")
    MISTRAL_MODEL: str = os.getenv("MISTRAL_MODEL", "mistral-small-latest")

    # OpenAI settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # Gemini settings
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

    MISTRAL_TEMPERATURE: float = 0.2  # Lower = more deterministic
    # Balanced mode defaults: faster responses with acceptable quality.
    MISTRAL_MAX_TOKENS_DEFAULT: int = int(os.getenv("MISTRAL_MAX_TOKENS_DEFAULT", "90"))
    MISTRAL_MAX_TOKENS_REFLECTION: int = int(
        os.getenv("MISTRAL_MAX_TOKENS_REFLECTION", "50")
    )


@dataclass
class ReflectionConfig:
    """Reflection (evaluation) configuration."""

    # Thresholds
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.60"))
    FACTUALITY_THRESHOLD: float = float(os.getenv("FACTUALITY_THRESHOLD", "0.80"))

    # Iteration limits
    MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "3"))

    # Runtime confidence weights (must align with ReflectiveAgent._compute_confidence)
    FACTUAL_CORRECTNESS_WEIGHT: float = float(
        os.getenv("FACTUAL_CORRECTNESS_WEIGHT", "0.45")
    )
    COMPLETENESS_WEIGHT: float = float(os.getenv("COMPLETENESS_WEIGHT", "0.30"))
    RELEVANCE_WEIGHT: float = float(os.getenv("RELEVANCE_WEIGHT", "0.25"))

    # Legacy fields kept for backward compatibility with older docs/scripts.
    COVERAGE_WEIGHT: float = 0.25
    FACTUALITY_WEIGHT: float = 0.35
    CONCISENESS_WEIGHT: float = 0.20
    COHERENCE_WEIGHT: float = 0.20


@dataclass
class ExcerptConfig:
    """Document excerpt extraction configuration."""

    EXCERPT_WINDOW_CHARS: int = 700  # Max characters per excerpt
    QUERY_CONTEXT_CHARS_BEFORE: int = int(700 / 3)
    QUERY_CONTEXT_CHARS_AFTER: int = int(700 * 2 / 3)


@dataclass
class ToolConfig:
    """External tool configuration."""

    # API keys
    SERPAPI_KEY: str = os.getenv("SERPAPI_KEY", "")
    WOLFRAM_ALPHA_KEY: str = os.getenv("WOLFRAM_ALPHA_KEY", "")

    # Behavior
    WEB_SEARCH_RESULTS_COUNT: int = int(os.getenv("WEB_SEARCH_RESULTS_COUNT", "3"))
    DUCKDUCKGO_TIMEOUT_SECONDS: int = int(os.getenv("DUCKDUCKGO_TIMEOUT_SECONDS", "10"))

    # Tools available
    AVAILABLE_TOOLS: list = None

    def __post_init__(self):
        if self.AVAILABLE_TOOLS is None:
            self.AVAILABLE_TOOLS = ["calculator", "web_search"]


@dataclass
class DatabaseConfig:
    """Persistent database and data-path settings."""

    BASE_DIR: str = os.path.dirname(__file__)
    CHROMA_DB_DIR: str = os.getenv(
        "CHROMA_DB_DIR", os.path.join(os.path.dirname(__file__), ".chroma_db")
    )
    CHROMADB_COLLECTION_NAME: str = os.getenv("CHROMADB_COLLECTION_NAME", "math_docs")
    CHROMADB_DISTANCE_METRIC: str = os.getenv("CHROMADB_DISTANCE_METRIC", "cosine")
    CHROMADB_ALLOW_RESET: bool = _bool_env("CHROMADB_ALLOW_RESET", False)
    CHROMADB_DISABLE_TELEMETRY: bool = _bool_env("CHROMADB_DISABLE_TELEMETRY", True)


@dataclass
class IngestionConfig:
    """Document ingestion settings."""

    DATA_DIR: str = os.path.join(os.path.dirname(__file__), "data")
    DEFAULT_JSONL_FILE: str = os.getenv("DEFAULT_JSONL_FILE", "sample_docs.jsonl")
    CHUNK_SIZE: int = int(os.getenv("INGEST_CHUNK_SIZE", "700"))
    CHUNK_OVERLAP: int = int(os.getenv("INGEST_CHUNK_OVERLAP", "100"))
    INGEST_BATCH_SIZE: int = int(os.getenv("INGEST_BATCH_SIZE", "100"))


@dataclass
class RuntimeConfig:
    """Runtime behavior toggles for demo/performance modes."""

    # FAST_MODE reduces latency by disabling expensive loops/LLM reflection.
    FAST_MODE: bool = _bool_env("FAST_MODE", False)
    # USE_CREWAI enables a thin CrewAI wrapper over the LangGraph execution path.
    USE_CREWAI: bool = _bool_env("USE_CREWAI", False)


# Global config instances (singleton pattern)
rag_config = RAGConfig()
llm_config = LLMConfig()
reflection_config = ReflectionConfig()
excerpt_config = ExcerptConfig()
tool_config = ToolConfig()
db_config = DatabaseConfig()
ingestion_config = IngestionConfig()
runtime_config = RuntimeConfig()
