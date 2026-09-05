from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from .models import DocumentChunk, SearchHit


class DocumentIngestor(Protocol):
    def discover(self, target: Path) -> list[Path]: ...

    def ingest_file(
        self,
        path: Path,
        *,
        content_hash: str,
        index_fingerprint: str,
    ) -> list[DocumentChunk]: ...


class EmbeddingProvider(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        hits: Sequence[SearchHit],
        limit: int,
    ) -> list[SearchHit]: ...


class VectorStore(Protocol):
    def count(self) -> int: ...

    def source_fingerprint(self, source: str) -> str | None: ...

    def list_sources(self) -> set[str]: ...

    def replace_source(
        self,
        source: str,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
        batch_size: int,
    ) -> None: ...

    def delete_sources(self, sources: Sequence[str]) -> None: ...

    def query_dense(
        self,
        query_embedding: Sequence[float],
        limit: int,
    ) -> list[SearchHit]: ...

    def get_all(self) -> list[SearchHit]: ...

    def get_by_ids(self, ids: Sequence[str]) -> list[SearchHit]: ...
