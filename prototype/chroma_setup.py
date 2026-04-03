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

    Note:
        - Uses PersistentClient (new Chroma API)
        - Data is stored in .chroma_db/ and persists across sessions
        - First call initializes; subsequent calls reuse existing DB
    """
    if persist_directory is None:
        persist_directory = CHROMA_DB_DIR

    # Ensure directory exists
    Path(persist_directory).mkdir(parents=True, exist_ok=True)

    try:
        # Create persistent client with the new Chroma API.
        settings = Settings(
            anonymized_telemetry=not db_config.CHROMADB_DISABLE_TELEMETRY,
            allow_reset=allow_reset or db_config.CHROMADB_ALLOW_RESET,
        )

        client = chromadb.PersistentClient(
            path=persist_directory,
            settings=settings,
        )
        logger.info(
            f"✓ ChromaDB client initialized (persisting to {persist_directory})"
        )
        return client

    except Exception as e:
        logger.error(f"✗ Failed to initialize ChromaDB client: {e}")
        raise


def get_chroma_collection(
    client: chromadb.Client,
    collection_name: str = db_config.CHROMADB_COLLECTION_NAME,
    distance_metric: str = db_config.CHROMADB_DISTANCE_METRIC,
) -> chromadb.Collection:
    """
    Get or create a ChromaDB collection for math documents.

    Args:
        client: ChromaDB client instance
        collection_name: Name of collection (e.g., "math_docs")
        distance_metric: Similarity metric: "cosine", "l2", or "ip"

    Returns:
        chromadb.Collection ready for adding/querying documents

    Raises:
        Exception: If collection creation fails
    """
    try:
        # Get or create collection with metadata
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={
                "description": "Collection of mathematical documents",
                "embedding_model": rag_config.EMBEDDING_MODEL,
                "distance_metric": distance_metric,
            },
            # Note: distance_metric is handled at query time in ChromaDB
        )
        logger.info(f"✓ Collection '{collection_name}' ready")
        return collection

    except Exception as e:
        logger.error(f"✗ Failed to get/create collection: {e}")
        raise


def get_embedding_model(model_name: str = rag_config.EMBEDDING_MODEL):
    """
    Load SentenceTransformer embedding model.

    Args:
        model_name: HuggingFace model name
                   Default: "all-MiniLM-L6-v2" (384 dims, fast)
                   Alternatives:
                     - "all-mpnet-base-v2" (768 dims, better quality)
                     - "paraphrase-MiniLM-L6-v2" (384 dims, specializ. paraphrase)

    Returns:
        SentenceTransformer model instance (cached for efficiency)
    """
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        logger.info(f"✓ Embedding model loaded: {model_name}")
        return model
    except ImportError:
        logger.error(
            "✗ sentence-transformers not installed. Run: pip install sentence-transformers"
        )
        raise
    except Exception as e:
        logger.error(f"✗ Failed to load embedding model {model_name}: {e}")
        raise


def check_collection_status(collection: chromadb.Collection) -> dict:
    """
    Check collection status: number of documents, metadata, etc.

    Args:
        collection: ChromaDB collection instance

    Returns:
        Status dictionary with document count and metadata
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
