"""
Document Ingestion Script: Load documents into ChromaDB with embeddings.

This script:
1. Loads documents from JSONL and PDFs
2. Chunks documents into smaller pieces
3. Generates embeddings using SentenceTransformer
4. Stores in ChromaDB with metadata

Usage:
    # Ingest all documents
    python ingest.py

    # Ingest and reset database
    python ingest.py --reset

    # Verbose output
    python ingest.py -v
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging

# Setup path to allow imports from repo root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import ChromaDB setup
from prototype.chroma_setup import (
    get_chroma_client,
    get_chroma_collection,
    get_embedding_model,
    check_collection_status,
    reset_collection,
)
from prototype.config import ingestion_config, db_config

# Data directory
DATA_PATH = ingestion_config.DATA_DIR

# Chunking settings
CHUNK_SIZE = ingestion_config.CHUNK_SIZE
CHUNK_OVERLAP = ingestion_config.CHUNK_OVERLAP


def load_jsonl_documents(filepath: str) -> List[Dict]:
    """
    Load documents from JSONL file.

    Args:
        filepath: Path to .jsonl file

    Returns:
        List of document dicts with fields: id, title, text, source, metadata
    """
    documents = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    doc = json.loads(line)
                    documents.append(doc)
                except json.JSONDecodeError as e:
                    logger.warning(f"  Line {line_num}: Invalid JSON - {e}")

        logger.info(f"✓ Loaded {len(documents)} documents from {filepath}")
        return documents

    except FileNotFoundError:
        logger.warning(f"✗ JSONL file not found: {filepath}")
        return []
    except Exception as e:
        logger.error(f"✗ Error loading JSONL: {e}")
        return []


def load_pdf_documents(data_directory: str) -> List[Dict]:
    """
    Load documents from PDF files in directory.

    Args:
        data_directory: Directory containing PDFs

    Returns:
        List of document dicts with fields: id, title, text, source
    """
    documents = []

    try:
        from pypdf import PdfReader

        pypdf_available = True
    except ImportError:
        logger.warning(
            "pypdf not installed. PDF loading disabled. Install with: pip install pypdf"
        )
        pypdf_available = False

    if not pypdf_available:
        return documents

    try:
        for fname in os.listdir(data_directory):
            if fname.lower().endswith(".pdf"):
                filepath = os.path.join(data_directory, fname)
                try:
                    reader = PdfReader(filepath)
                    pages = []

                    for page_num, page in enumerate(reader.pages, 1):
                        text = page.extract_text() or ""
                        if text.strip():
                            pages.append({"page": page_num, "text": text.strip()})

                    # Combine all pages
                    full_text = "\n".join([p["text"] for p in pages])

                    doc = {
                        "id": os.path.splitext(fname)[0],
                        "title": os.path.splitext(fname)[0],
                        "text": full_text,
                        "source": "pdf",
                        "metadata": {
                            "pages": len(pages),
                            "filename": fname,
                        },
                    }

                    documents.append(doc)
                    logger.info(f"  ✓ Loaded PDF: {fname} ({len(pages)} pages)")

                except Exception as e:
                    logger.warning(f"  ✗ Failed to read PDF {fname}: {e}")

        logger.info(f"✓ Loaded {len(documents)} PDFs from {data_directory}")
        return documents

    except Exception as e:
        logger.error(f"✗ Error loading PDFs: {e}")
        return []


def chunk_document(
    doc: Dict, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> List[Dict]:
    """
    Split a document into overlapping chunks for better retrieval.

    Args:
        doc: Document dict with 'text' field
        chunk_size: Characters per chunk
        overlap: Characters of overlap between chunks

    Returns:
        List of chunk dicts with metadata pointing to original document
    """
    text = doc.get("text", "")

    # Adaptive chunk sizing: larger PDFs use bigger chunks to reduce fragment count
    if len(text) > 100000:  # Large doc (e.g., 96-page PDF)
        chunk_size = max(chunk_size, 1200)  # Use larger chunks
        overlap = min(overlap, 150)  # Reduce overlap for faster processing

    if not text or len(text) <= chunk_size:
        # Document fits in single chunk
        return [
            {
                "chunk_id": f"{doc.get('id', 'doc')}_0",
                "text": text,
                "chunk_num": 0,
                "original_id": doc.get("id", "unknown"),
                "title": doc.get("title", "Unknown"),
                "source": doc.get("source", "unknown"),
                "metadata": doc.get("metadata", {}),
            }
        ]

    # Normalize overlap to guarantee forward progress even with bad env values.
    overlap = max(0, min(overlap, chunk_size - 1))
    step = max(1, chunk_size - overlap)

    # Split into overlapping chunks - simple and fast algorithm
    chunks = []
    start = 0
    chunk_num = 0
    doc_length = len(text)

    while start < doc_length:
        # Calculate end position
        end = min(start + chunk_size, doc_length)
        chunk_text = text[start:end].strip()

        if (
            chunk_text and len(chunk_text) > 50
        ):  # Only keep chunks with meaningful content
            chunks.append(
                {
                    "chunk_id": f"{doc.get('id', 'doc')}_{chunk_num}",
                    "text": chunk_text,
                    "chunk_num": chunk_num,
                    "original_id": doc.get("id", "unknown"),
                    "title": doc.get("title", "Unknown"),
                    "source": doc.get("source", "unknown"),
                    "metadata": doc.get("metadata", {}),
                }
            )
            chunk_num += 1

        # Heartbeat log for very large docs so users can see progress.
        if chunk_num > 0 and chunk_num % 100 == 0:
            pct = (end / doc_length) * 100
            logger.info(f"    ... chunking progress: {chunk_num} chunks ({pct:.1f}%)")

        # Move to next chunk with guaranteed forward progress.
        start += step

    return chunks


def chunk_documents(documents: List[Dict]) -> List[Dict]:
    """
    Chunk all documents.

    Args:
        documents: List of document dicts

    Returns:
        List of all chunks from all documents
    """
    all_chunks = []

    for idx, doc in enumerate(documents, 1):
        logger.info(
            f"  Chunking document {idx}/{len(documents)}: {doc.get('title', 'Unknown')}..."
        )
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
        logger.info(f"    → Created {len(chunks)} chunks")

    logger.info(f"✓ Created {len(all_chunks)} chunks from {len(documents)} documents")
    return all_chunks


def ingest_into_chromadb(
    documents: List[Dict],
    collection_name: str = db_config.CHROMADB_COLLECTION_NAME,
    batch_size: int = ingestion_config.INGEST_BATCH_SIZE,
    reset: bool = False,
) -> Tuple[int, str]:
    """
    Ingest documents into ChromaDB with embeddings.

    Args:
        documents: List of document chunks
        collection_name: Target collection name
        batch_size: Number of docs per batch (for memory efficiency)
        reset: If True, delete existing collection first

    Returns:
        Tuple of (number_ingested, status_message)
    """
    # Initialize ChromaDB
    logger.info("\n🗄️  Initializing ChromaDB...")
    client = get_chroma_client(allow_reset=reset)

    # Reset collection if requested
    if reset:
        logger.info("🔄 Resetting collection (deleting existing data)...")
        collection = reset_collection(client, collection_name)
    else:
        collection = get_chroma_collection(client, collection_name)

    # Load embedding model
    logger.info("\n📦 Loading embedding model...")
    model = get_embedding_model()

    # Prepare ingestion
    logger.info(f"\n📝 Ingesting {len(documents)} chunks into '{collection_name}'...")

    ids = []
    texts = []
    metadatas = []
    total_ingested = 0

    # Process documents in batches
    for i, doc in enumerate(documents):
        ids.append(doc["chunk_id"])
        texts.append(doc["text"])
        metadatas.append(
            {
                "original_id": doc["original_id"],
                "title": doc["title"],
                "source": doc["source"],
                "chunk_num": str(doc["chunk_num"]),
            }
        )

        # Process batch
        if (i + 1) % batch_size == 0 or i == len(documents) - 1:
            batch_num = (i // batch_size) + 1
            total_batches = (len(documents) + batch_size - 1) // batch_size
            logger.info(
                f"  Batch {batch_num}/{total_batches}: embedding {len(texts)} chunks..."
            )

            # Generate embeddings for batch (with progress)
            try:
                batch_embeddings = model.encode(texts, show_progress_bar=False).tolist()

                # Add batch to ChromaDB
                collection.add(
                    ids=ids,
                    embeddings=batch_embeddings,
                    documents=texts,
                    metadatas=metadatas,
                )
                total_ingested += len(ids)
                logger.info(
                    f"    ✓ Stored {len(ids)} chunks (total ingested: {total_ingested})"
                )

            except Exception as e:
                logger.error(f"    ✗ Failed to ingest batch: {e}")
                return total_ingested, f"Error at batch {batch_num}: {str(e)}"

            # Reset batch
            ids = []
            texts = []
            metadatas = []

    # Verify ingestion
    status = check_collection_status(collection)

    logger.info(f"\n✅ Ingestion complete!")
    logger.info(f"   Documents stored: {status['document_count']}")

    return status["document_count"], "Successfully ingested"


def main():
    """Main ingestion pipeline."""
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB")
    parser.add_argument(
        "--reset", action="store_true", help="Reset collection before ingesting"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=" * 60)
    logger.info("📚 ChromaDB Document Ingestion Pipeline")
    logger.info("=" * 60)

    # Load documents
    logger.info("\n📂 Loading documents...")
    documents = []

    # Load JSONL
    jsonl_path = os.path.join(DATA_PATH, ingestion_config.DEFAULT_JSONL_FILE)
    documents.extend(load_jsonl_documents(jsonl_path))

    # Load PDFs
    documents.extend(load_pdf_documents(DATA_PATH))

    if not documents:
        logger.warning("⚠️  No documents found! Checked:")
        logger.warning(f"   - JSONL: {jsonl_path}")
        logger.warning(f"   - PDFs: {DATA_PATH}")
        logger.warning("   Please add documents and run again.")
        return

    # Chunk documents
    logger.info("\n✂️  Chunking documents...")
    chunks = chunk_documents(documents)

    # Ingest into ChromaDB
    try:
        num_ingested, status_msg = ingest_into_chromadb(
            chunks,
            reset=args.reset,
        )
        logger.info(f"✓ {status_msg}: {num_ingested} chunks")
    except Exception as e:
        logger.error(f"✗ Ingestion failed: {e}", exc_info=True)
        raise

    logger.info("=" * 60)
    logger.info("✓ Done! Documents are ready for retrieval.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
