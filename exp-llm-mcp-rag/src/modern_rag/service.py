from __future__ import annotations

import logging
import threading
import time
from collections.abc import Sequence
from pathlib import Path

from .config import RAGSettings
from .contracts import DocumentIngestor, EmbeddingProvider, Reranker, VectorStore
from .ingestion import file_content_hash
from .models import IndexFailure, IndexReport, SearchHit
from .retrieval import HybridRetriever

_logger = logging.getLogger(__name__)


class KnowledgeRAG:
    """Thread-safe indexing and retrieval; the outer Agent remains the generator."""

    def __init__(
        self,
        settings: RAGSettings,
        ingestor: DocumentIngestor,
        embeddings: EmbeddingProvider,
        reranker: Reranker,
        store: VectorStore,
    ) -> None:
        self.settings = settings
        self.ingestor = ingestor
        self.embeddings = embeddings
        self.reranker = reranker
        self.store = store
        self._retriever: HybridRetriever | None = None
        self._lock = threading.RLock()

    def _embed_in_batches(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        batch_size = self.settings.embedding_batch_size
        for start in range(0, len(texts), batch_size):
            vectors.extend(self.embeddings.embed(texts[start : start + batch_size]))
        return vectors

    def _new_retriever(self) -> HybridRetriever:
        return HybridRetriever(
            store=self.store,
            embeddings=self.embeddings,
            reranker=self.reranker,
            dense_k=self.settings.dense_k,
            keyword_k=self.settings.keyword_k,
            fusion_k=self.settings.fusion_k,
            final_k=self.settings.final_k,
            rank_constant=self.settings.rrf_constant,
            jieba_user_dict=self.settings.jieba_user_dict,
        )

    @staticmethod
    def _sources_missing_from_directory(
        indexed_sources: set[str],
        root: Path,
        current_sources: set[str],
    ) -> list[str]:
        missing: list[str] = []
        for source in indexed_sources:
            try:
                belongs_to_root = Path(source).resolve().is_relative_to(root)
            except (OSError, ValueError):
                belongs_to_root = False
            if belongs_to_root and source not in current_sources:
                missing.append(source)
        return missing

    def index(self, target: str | Path) -> IndexReport:
        started = time.perf_counter()
        target_path = Path(target).expanduser().resolve()
        files = self.ingestor.discover(target_path)
        report = IndexReport(discovered_files=len(files))

        with self._lock:
            index_changed = False
            current_sources = {str(path.resolve()) for path in files}
            for path in files:
                source = str(path.resolve())
                try:
                    content_hash = file_content_hash(path)
                    fingerprint = self.settings.index_fingerprint(content_hash)
                    if self.store.source_fingerprint(source) == fingerprint:
                        report.skipped_files += 1
                        continue

                    chunks = self.ingestor.ingest_file(
                        path,
                        content_hash=content_hash,
                        index_fingerprint=fingerprint,
                    )
                    vectors = self._embed_in_batches(
                        [chunk.embedding_text for chunk in chunks]
                    )
                    self.store.replace_source(
                        source,
                        chunks,
                        vectors,
                        batch_size=self.settings.embedding_batch_size,
                    )
                    report.indexed_files += 1
                    report.indexed_chunks += len(chunks)
                    index_changed = True
                except Exception as exc:  # noqa: BLE001
                    _logger.exception("RAG source indexing failed: %s", source)
                    report.failures.append(
                        IndexFailure(
                            source=source,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    )

            if target_path.is_dir() and self.settings.prune_missing_sources:
                missing = self._sources_missing_from_directory(
                    self.store.list_sources(),
                    target_path,
                    current_sources,
                )
                if missing:
                    self.store.delete_sources(missing)
                    report.removed_sources = len(missing)
                    index_changed = True

            if index_changed:
                self._retriever = self._new_retriever()

        report.elapsed_ms = round((time.perf_counter() - started) * 1000)
        _logger.info("RAG indexing completed: %s", report)
        return report

    def search(self, query: str, limit: int | None = None) -> list[SearchHit]:
        query = query.strip()
        if not query:
            raise ValueError("检索问题不能为空")
        if limit is not None and limit <= 0:
            raise ValueError("limit 必须大于 0")

        with self._lock:
            if self.store.count() == 0:
                raise RuntimeError("RAG 索引为空，请先调用 index_documents()")
            if self._retriever is None:
                self._retriever = self._new_retriever()
            return self._retriever.retrieve(query, limit)
