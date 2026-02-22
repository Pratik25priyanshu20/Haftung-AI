"""StVO/case law ingestion into Qdrant."""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path

from haftung_ai.config.settings import get_settings
from haftung_ai.rag.embeddings import EmbeddingService
from haftung_ai.rag.vectorstore import VectorStore

logger = logging.getLogger(__name__)


class KnowledgeBaseIngester:
    """Ingest legal documents (StVO, case law, insurance guidelines) into Qdrant."""

    def __init__(self):
        self.embedder = EmbeddingService()
        self.vectorstore = VectorStore()
        self.vectorstore.ensure_collection(self.embedder.dimension)

    def ingest_directory(self, path: str | Path, source_name: str = "unknown") -> int:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        settings = get_settings()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " "],
        )

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Knowledge base path not found: {path}")

        chunks: list[dict] = []
        for file in path.rglob("*"):
            if file.suffix.lower() not in (".txt", ".md", ".pdf"):
                continue
            text = self._read_file(file)
            if not text:
                continue
            splits = splitter.split_text(text)
            for idx, split in enumerate(splits):
                chunk_id = str(uuid.uuid4())
                chunks.append({
                    "chunk_id": chunk_id,
                    "content": split,
                    "metadata": {
                        "source": str(file),
                        "source_name": source_name,
                        "file_name": file.name,
                        "chunk_index": idx,
                        "created_at": datetime.utcnow().isoformat(),
                    },
                })

        if not chunks:
            return 0

        texts = [c["content"] for c in chunks]
        embeddings = self.embedder.embed_documents(texts)
        self.vectorstore.upsert_chunks(chunks, embeddings)

        logger.info("Ingested %d chunks from %s", len(chunks), path)
        return len(chunks)

    def _read_file(self, path: Path) -> str:
        if path.suffix.lower() == ".pdf":
            try:
                from pypdf import PdfReader

                reader = PdfReader(str(path))
                return "\n".join(page.extract_text() or "" for page in reader.pages)
            except Exception as e:
                logger.warning("Failed to read PDF %s: %s", path, e)
                return ""
        else:
            try:
                return path.read_text(encoding="utf-8")
            except Exception as e:
                logger.warning("Failed to read %s: %s", path, e)
                return ""
