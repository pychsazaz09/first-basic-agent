from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .models import DocumentChunk, Metadata, SearchHit


def _clean_metadata(value: object) -> Metadata:
    if not isinstance(value, dict):
        return {}
    allowed = (str, int, float, bool)
    return {
        str(key): item
        for key, item in value.items()
        if isinstance(item, allowed)
    }


class ChromaVectorStore:
    """Persistent cosine vector store with version-safe source replacement."""

    def __init__(
        self,
        persist_dir: Path,
        collection_name: str,
        embedding_model: str,
        index_schema_version: str,
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("缺少 chromadb") from exc

        persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        expected_metadata = {
            "description": "Docling hybrid retrieval for RAGAgent",
            "embedding_model": embedding_model,
            "index_schema_version": index_schema_version,
        }
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
            metadata=expected_metadata,
        )
        actual = self.collection.metadata or {}
        hnsw_configuration = self.collection.configuration.get("hnsw") or {}
        if (
            actual.get("embedding_model") != embedding_model
            or str(actual.get("index_schema_version")) != index_schema_version
            or hnsw_configuration.get("space") != "cosine"
        ):
            raise RuntimeError(
                f"Chroma collection {collection_name!r} 与当前 embedding/index schema "
                "不兼容；请使用新的 RAG_CHROMA_COLLECTION 并重新索引"
            )

    def count(self) -> int:
        return int(self.collection.count())

    def source_fingerprint(self, source: str) -> str | None:
        result = self.collection.get(
            where={"source": source},
            include=["metadatas"],
        )
        metadatas = [
            _clean_metadata(value)
            for value in result.get("metadatas") or []
        ]
        if not metadatas:
            return None
        fingerprints = {
            str(metadata.get("index_fingerprint", ""))
            for metadata in metadatas
        }
        expected_counts = {
            int(metadata.get("source_chunk_count", 0))
            for metadata in metadatas
        }
        if (
            len(fingerprints) != 1
            or "" in fingerprints
            or len(expected_counts) != 1
            or expected_counts != {len(metadatas)}
        ):
            return None
        return fingerprints.pop()

    def list_sources(self) -> set[str]:
        result = self.collection.get(include=["metadatas"])
        return {
            str(metadata["source"])
            for value in result.get("metadatas") or []
            if (metadata := _clean_metadata(value)).get("source")
        }

    def replace_source(
        self,
        source: str,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
        batch_size: int,
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("文本块数量与向量数量不一致")
        if not chunks:
            raise ValueError("不能写入空文档")
        if any(chunk.source != source for chunk in chunks):
            raise ValueError("一次 replace_source 只能写入同一个源文件")

        old_result = self.collection.get(
            where={"source": source},
            include=["metadatas"],
        )
        old_ids = {str(value) for value in old_result.get("ids") or []}
        new_ids = {chunk.id for chunk in chunks}

        try:
            for start in range(0, len(chunks), batch_size):
                chunk_batch = list(chunks[start : start + batch_size])
                embedding_batch = embeddings[start : start + batch_size]
                self.collection.upsert(
                    ids=[chunk.id for chunk in chunk_batch],
                    documents=[chunk.text for chunk in chunk_batch],
                    metadatas=[chunk.metadata for chunk in chunk_batch],
                    embeddings=[list(vector) for vector in embedding_batch],
                )
        except Exception:
            self.collection.delete(ids=list(new_ids))
            raise

        stale_ids = old_ids - new_ids
        if stale_ids:
            self.collection.delete(ids=list(stale_ids))

    def delete_sources(self, sources: Sequence[str]) -> None:
        for source in sorted(set(sources)):
            self.collection.delete(where={"source": source})

    def query_dense(
        self,
        query_embedding: Sequence[float],
        limit: int,
    ) -> list[SearchHit]:
        if self.count() == 0:
            return []
        result = self.collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=min(limit, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        return [
            SearchHit(
                id=str(item_id),
                text=str(document or ""),
                metadata=_clean_metadata(metadata),
                dense_rank=rank,
                dense_distance=float(distance),
            )
            for rank, (item_id, document, metadata, distance) in enumerate(
                zip(ids, documents, metadatas, distances),
                start=1,
            )
        ]

    def get_all(self) -> list[SearchHit]:
        result: dict[str, Any] = self.collection.get(
            include=["documents", "metadatas"]
        )
        return [
            SearchHit(
                id=str(item_id),
                text=str(document or ""),
                metadata=_clean_metadata(metadata),
            )
            for item_id, document, metadata in zip(
                result.get("ids") or [],
                result.get("documents") or [],
                result.get("metadatas") or [],
            )
        ]

    def get_by_ids(self, ids: Sequence[str]) -> list[SearchHit]:
        if not ids:
            return []
        result: dict[str, Any] = self.collection.get(
            ids=list(ids),
            include=["documents", "metadatas"],
        )
        found = {
            str(item_id): SearchHit(
                id=str(item_id),
                text=str(document or ""),
                metadata=_clean_metadata(metadata),
            )
            for item_id, document, metadata in zip(
                result.get("ids") or [],
                result.get("documents") or [],
                result.get("metadatas") or [],
            )
        }
        return [found[item_id] for item_id in ids if item_id in found]
