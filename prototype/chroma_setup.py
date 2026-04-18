"""
ChromaDB Setup Module: Initialize persistent vector database.

This module handles:
1. Persistent ChromaDB client initialization
2. Collection creation and management
3. Embedding model configuration
4. Database health checks

Usage:
    from prototype.chroma_setup import get_chroma_client, get_chroma_collection

    client = get_chroma_client()
    collection = get_chroma_collection(client)
    collection.query(query_embeddings=[[...]], n_results=5)
"""

import logging
import shutil
import os
import glob
from datetime import datetime
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from prototype.config import db_config, rag_config

logger = logging.getLogger(__name__)

# ChromaDB persistence directory (persistent across runs)
CHROMA_DB_DIR = db_config.CHROMA_DB_DIR

# Ensure directory exists
Path(CHROMA_DB_DIR).mkdir(parents=True, exist_ok=True)

# Module-level cache for the embedding model to avoid repeated downloads/loads
_EMBEDDING_MODEL = None


def _is_fatal_base_exception(exc: BaseException) -> bool:
    """Return True for control-flow exceptions that should never be swallowed."""
    return isinstance(exc, (KeyboardInterrupt, SystemExit, GeneratorExit))


def _backup_corrupt_chroma_dir(persist_directory: str) -> Optional[Path]:
    """Move a suspected-corrupt Chroma directory aside and recreate it."""
    db_dir = Path(persist_directory)
    if not db_dir.exists():
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = db_dir.with_name(f"{db_dir.name}_corrupt_{timestamp}")
    shutil.move(str(db_dir), str(backup_dir))
    db_dir.mkdir(parents=True, exist_ok=True)
    return backup_dir


def get_chroma_client(
    persist_directory: Optional[str] = None, allow_reset: bool = False
) -> chromadb.Client:
    """
    Initialize persistent ChromaDB client.

    Args:
        persist_directory: Path to persistence directory
                          Defaults to .chroma_db/ in prototype/
        allow_reset: If True, allow resetting the database via API

    Returns:
        ChromaDB client with persistent storage
    """
    if persist_directory is None:
        persist_directory = CHROMA_DB_DIR

    # Ensure directory exists
    Path(persist_directory).mkdir(parents=True, exist_ok=True)

    # Create persistent client with the new Chroma API.
    settings = Settings(
        anonymized_telemetry=not db_config.CHROMADB_DISABLE_TELEMETRY,
        allow_reset=allow_reset or db_config.CHROMADB_ALLOW_RESET,
    )

    def _build_persistent_client() -> chromadb.Client:
        return chromadb.PersistentClient(
            path=persist_directory,
            settings=settings,
        )

    try:
        client = _build_persistent_client()
        logger.info(
            f"✓ ChromaDB client initialized (persisting to {persist_directory})"
        )
        return client

    except BaseException as e:
        if _is_fatal_base_exception(e):
            raise
        logger.error(f"✗ Failed to initialize ChromaDB client: {e}")

        try:
            backup_dir = _backup_corrupt_chroma_dir(persist_directory)
            if backup_dir is not None:
                logger.warning(
                    "Moved possibly-corrupt Chroma directory to '%s'; retrying fresh DB.",
                    backup_dir,
                )
                client = _build_persistent_client()
                logger.info(
                    "✓ ChromaDB client initialized after recovery (persisting to %s)",
                    persist_directory,
                )
                return client
        except BaseException as retry_exc:
            if _is_fatal_base_exception(retry_exc):
                raise
            logger.error("✗ ChromaDB recovery retry failed: %s", retry_exc)

        # Compatibility fallback: older client initialization path can work on
        # some environments where the rust-backed PersistentClient fails.
        try:
            legacy_settings = Settings(
                is_persistent=True,
                persist_directory=persist_directory,
                anonymized_telemetry=not db_config.CHROMADB_DISABLE_TELEMETRY,
                allow_reset=allow_reset or db_config.CHROMADB_ALLOW_RESET,
            )
            client = chromadb.Client(settings=legacy_settings)
            logger.warning(
                "ChromaDB initialized via legacy client compatibility mode at %s",
                persist_directory,
            )
            return client
        except BaseException as legacy_exc:
            if _is_fatal_base_exception(legacy_exc):
                raise
            logger.error("✗ ChromaDB legacy compatibility mode failed: %s", legacy_exc)

        raise RuntimeError(f"ChromaDB initialization failed: {e}") from None


def get_chroma_collection(
    client: chromadb.Client,
    collection_name: str = db_config.CHROMADB_COLLECTION_NAME,
    distance_metric: str = db_config.CHROMADB_DISTANCE_METRIC,
) -> chromadb.Collection:
    """
    Get or create a ChromaDB collection for math documents.
    """
    try:
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={
                "description": "Collection of mathematical documents",
                "embedding_model": rag_config.EMBEDDING_MODEL,
                "distance_metric": distance_metric,
            },
        )
        logger.info(f"✓ Collection '{collection_name}' ready")
        return collection

    except Exception as e:
        logger.error(f"✗ Failed to get/create collection: {e}")
        raise


def get_embedding_model(model_name: str = rag_config.EMBEDDING_MODEL):
    """
    Load SentenceTransformer embedding model with preference for local cache.
    """
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is not None:
        return _EMBEDDING_MODEL

    try:
        from sentence_transformers import SentenceTransformer

        # Prefer a locally cached model directory if provided via environment.
        model_dir = os.environ.get("MODEL_CACHE_DIR")
        hf_home = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")

        # Log the environment context so we can diagnose remote HEAD retries.
        logger.info(
            "Embedding model load context: MODEL_CACHE_DIR=%s HF_HOME=%s model_name=%s",
            model_dir,
            hf_home,
            model_name,
        )

        candidates = []
        if model_dir:
            candidates.append(model_dir)
        if hf_home:
            candidates.append(os.path.join(hf_home, model_name))
            owner_repo = model_name.replace("/", "--")
            pattern = os.path.join(hf_home, f"models--{owner_repo}*")
            matches = glob.glob(pattern)
            candidates.extend(sorted(matches))

        # Also probe some common local cache locations (useful on Windows/USB drives)
        home = Path.home()
        common_paths = [
            os.path.join("A:", "hf_cache", model_name),
            os.path.join(
                str(home),
                ".cache",
                "huggingface",
                "hub",
                "models--" + model_name.replace("/", "--"),
            ),
            os.path.join(str(home), ".cache", "huggingface", "hub", model_name),
        ]
        for p in common_paths:
            candidates.append(p)

        # Expand and search for likely candidate directories
        expanded = []
        for cand in candidates:
            if not cand:
                continue
            p = Path(cand)
            if p.exists():
                expanded.append(str(p))
            else:
                glob_pattern = str(p) + "*"
                matches = glob.glob(glob_pattern)
                expanded.extend(matches)

        if hf_home:
            token = model_name.split("/")[-1]
            for match in glob.glob(
                os.path.join(hf_home, "**", f"*{token}*"), recursive=True
            ):
                expanded.append(match)

        # Deduplicate
        seen = set()
        candidates_to_try = []
        for c in expanded:
            if c and c not in seen:
                seen.add(c)
                candidates_to_try.append(c)

        logger.debug("Embedding model candidate paths: %s", candidates_to_try)

        for cand in candidates_to_try:
            try:
                cand_path = Path(cand)
                if not cand_path.exists() or not cand_path.is_dir():
                    continue
                # Check for common model files
                has_model_file = any(
                    (cand_path / fname).exists()
                    for fname in (
                        "model.safetensors",
                        "pytorch_model.bin",
                        "tf_model.h5",
                    )
                )
                if not has_model_file:
                    for sub in cand_path.iterdir():
                        if sub.is_dir() and any(
                            (sub / fname).exists()
                            for fname in (
                                "model.safetensors",
                                "pytorch_model.bin",
                                "tf_model.h5",
                            )
                        ):
                            cand_path = sub
                            has_model_file = True
                            break
                if not has_model_file:
                    continue

                _EMBEDDING_MODEL = SentenceTransformer(str(cand_path))
                logger.info("✓ Embedding model loaded from local cache: %s", cand_path)
                return _EMBEDDING_MODEL
            except Exception:
                logger.debug(
                    "Failed to load embedding from candidate: %s", cand, exc_info=True
                )

        # Fallback: load by model name (may download from Hugging Face)
        _EMBEDDING_MODEL = SentenceTransformer(model_name)
        logger.info(f"✓ Embedding model loaded: {model_name}")
        return _EMBEDDING_MODEL
    except ImportError as e:
        logger.error(
            "✗ Failed to import sentence-transformers dependencies: %s. "
            "Try reinstalling from prototype/requirements.txt.",
            e,
        )
        raise
    except Exception as e:
        logger.error(f"✗ Failed to load embedding model {model_name}: {e}")
        raise


def check_collection_status(collection: chromadb.Collection) -> dict:
    """
    Check collection status: number of documents, metadata, etc.
    """
    try:
        count = collection.count()
        metadata = collection.metadata

        status = {
            "collection_name": collection.name,
            "document_count": count,
            "metadata": metadata,
            "status": "healthy" if count > 0 else "empty",
        }

        logger.info(f"Collection status: {count} documents")
        return status

    except Exception as e:
        logger.error(f"✗ Failed to check collection status: {e}")
        return {"status": "error", "error": str(e)}


def reset_collection(
    client: chromadb.Client,
    collection_name: str = db_config.CHROMADB_COLLECTION_NAME,
):
    """
    Delete and recreate a collection (use with caution!).

    Args:
        client: ChromaDB client
        collection_name: Name of collection to reset

    Returns:
        New collection instance
    """
    try:
        client.delete_collection(name=collection_name)
        logger.warning(f"🗑️  Deleted collection '{collection_name}'")

        # Recreate empty collection
        new_collection = get_chroma_collection(client, collection_name)
        logger.info(f"✓ Created new empty collection '{collection_name}'")
        return new_collection

    except Exception as e:
        logger.error(f"✗ Failed to reset collection: {e}")
        raise


if __name__ == "__main__":
    # Demo: Initialize and check database
    logging.basicConfig(level=logging.INFO)

    client = get_chroma_client()
    collection = get_chroma_collection(client)
    status = check_collection_status(collection)

    print(f"\n📊 Collection Status:")
    print(f"  Name: {status['collection_name']}")
    print(f"  Documents: {status['document_count']}")
    print(f"  Status: {status['status']}")
    print(f"\n✓ ChromaDB is ready to use!")
