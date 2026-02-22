"""Ingest StVO, case law, and insurance documents into the vector store."""
from __future__ import annotations

import logging
from pathlib import Path

from haftung_ai.config.settings import get_settings
from haftung_ai.rag.knowledge_base import KnowledgeBaseIngester

logger = logging.getLogger(__name__)


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Ingest knowledge base into Qdrant")
    parser.add_argument(
        "--kb-dir",
        type=Path,
        default=None,
        help="Knowledge base directory (default: from settings)",
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--recreate", action="store_true", help="Delete and recreate collection")
    args = parser.parse_args()

    settings = get_settings()
    kb_dir = args.kb_dir or settings.KNOWLEDGE_BASE_DIR

    if not kb_dir.exists():
        logger.error("Knowledge base directory does not exist: %s", kb_dir)
        return

    ingester = KnowledgeBaseIngester()

    if args.recreate:
        logger.info("Recreating vector store collection...")
        ingester.vectorstore.client.delete_collection(settings.QDRANT_COLLECTION)

    # Ingest each source type
    source_types = ["stvo", "case_law", "insurance_guidelines"]
    total_chunks = 0

    for source_type in source_types:
        source_dir = kb_dir / source_type
        if not source_dir.exists():
            logger.warning("Source directory not found: %s", source_dir)
            continue

        logger.info("Ingesting %s from %s...", source_type, source_dir)
        chunks = ingester.ingest_directory(
            source_dir,
            source_type=source_type,
            batch_size=args.batch_size,
        )
        total_chunks += chunks
        logger.info("  Ingested %d chunks from %s", chunks, source_type)

    logger.info("Total: %d chunks ingested", total_chunks)


if __name__ == "__main__":
    main()
