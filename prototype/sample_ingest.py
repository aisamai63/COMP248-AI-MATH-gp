"""
Sample ingestion helper: ingest a small subset of chunks into ChromaDB
Used for performance testing of retrieval with a tiny collection.
"""

import argparse
import logging
from pathlib import Path

from prototype.ingest import load_jsonl_documents, chunk_documents, ingest_into_chromadb
from prototype.config import ingestion_config, db_config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DATA_PATH = ingestion_config.DATA_DIR


def main():
    parser = argparse.ArgumentParser(description="Ingest a small sample into ChromaDB")
    parser.add_argument(
        "-n", "--num", type=int, default=10, help="Number of chunks to ingest"
    )
    parser.add_argument(
        "--reset", action="store_true", help="Reset collection before ingest"
    )
    args = parser.parse_args()

    logger.info("Loading documents from %s", DATA_PATH)
    docs = load_jsonl_documents(Path(DATA_PATH))
    if not docs:
        logger.error("No documents found in %s", DATA_PATH)
        return

    logger.info("Chunking documents (this may take a moment)")
    chunks = chunk_documents(docs)
    sample = chunks[: args.num]
    logger.info("Ingesting %d chunks (reset=%s)", len(sample), args.reset)

    num_ingested, status = ingest_into_chromadb(sample, reset=args.reset)
    logger.info("Result: %s (%d chunks)", status, num_ingested)


if __name__ == "__main__":
    main()
