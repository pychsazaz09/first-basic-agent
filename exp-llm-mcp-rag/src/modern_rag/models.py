from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeAlias

INDEX_SCHEMA_VERSION = "2"

MetadataValue: TypeAlias = str | int | float | bool
Metadata: TypeAlias = dict[str, MetadataValue]


@dataclass(frozen=True, slots=True)
class SourceLocator:
    source: str
    source_name: str
    file_type: str
    chunk_index: int
    heading_path: tuple[str, ...] = ()
    page_start: int | None = None
    page_end: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    element_refs: tuple[str, ...] = ()
    element_labels: tuple[str, ...] = ()
    provenance_json: str = "[]"

    @property
    def location(self) -> str:
        parts: list[str] = []
        if self.page_start is not None:
            page_end = self.page_end or self.page_start
            parts.append(
                f"第 {self.page_start} 页"
                if page_end == self.page_start
                else f"第 {self.page_start}–{page_end} 页"
            )
        if self.heading_path:
            parts.append(" / ".join(self.heading_path))
        if self.line_start is not None:
            line_end = self.line_end or self.line_start
            parts.append(
                f"第 {self.line_start} 行"
                if line_end == self.line_start
                else f"第 {self.line_start}–{line_end} 行"
            )
        if self.element_labels and self.file_type == "docx":
            parts.append("、".join(self.element_labels))
        return " · ".join(parts) or f"结构块 {self.chunk_index}"

    @property
    def citation(self) -> str:
        return f"【{self.source_name}｜{self.location}】"

    def to_metadata(
        self,
        *,
        content_hash: str,
        index_fingerprint: str,
    ) -> Metadata:
        metadata: Metadata = {
            "source": self.source,
            "source_name": self.source_name,
            "file_type": self.file_type,
            "chunk_index": self.chunk_index,
            "location": self.location,
            "citation": self.citation,
            "content_hash": content_hash,
            "index_fingerprint": index_fingerprint,
            "index_schema_version": INDEX_SCHEMA_VERSION,
            "provenance": self.provenance_json,
        }
        if self.heading_path:
            metadata["heading_path"] = json.dumps(
                self.heading_path,
                ensure_ascii=False,
            )
        if self.page_start is not None:
            metadata["page_start"] = self.page_start
            metadata["page_end"] = self.page_end or self.page_start
        if self.line_start is not None:
            metadata["line_start"] = self.line_start
            metadata["line_end"] = self.line_end or self.line_start
        if self.element_refs:
            metadata["element_refs"] = json.dumps(self.element_refs)
        if self.element_labels:
            metadata["element_labels"] = json.dumps(
                self.element_labels,
                ensure_ascii=False,
            )
        return metadata


@dataclass(slots=True)
class DocumentChunk:
    id: str
    text: str
    embedding_text: str
    source: str
    location: str
    metadata: Metadata = field(default_factory=dict)


@dataclass(slots=True)
class SearchHit:
    id: str
    text: str
    metadata: Metadata
    dense_rank: int | None = None
    dense_distance: float | None = None
    keyword_rank: int | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None

    @property
    def source(self) -> str:
        return str(self.metadata.get("source", "未知来源"))

    @property
    def source_name(self) -> str:
        value = self.metadata.get("source_name")
        return str(value) if value else Path(self.source).name

    @property
    def location(self) -> str:
        return str(self.metadata.get("location", "未知位置"))

    @property
    def citation(self) -> str:
        value = self.metadata.get("citation")
        if value:
            return str(value)
        return f"【{self.source_name}｜{self.location}】"


@dataclass(frozen=True, slots=True)
class IndexFailure:
    source: str
    error: str


@dataclass(slots=True)
class IndexReport:
    discovered_files: int = 0
    indexed_files: int = 0
    skipped_files: int = 0
    removed_sources: int = 0
    indexed_chunks: int = 0
    failures: list[IndexFailure] = field(default_factory=list)
    elapsed_ms: int = 0

    @property
    def ok(self) -> bool:
        return not self.failures

    def __str__(self) -> str:
        return (
            f"发现 {self.discovered_files} 个文件，索引 {self.indexed_files} 个，"
            f"跳过 {self.skipped_files} 个，移除 {self.removed_sources} 个旧源，"
            f"写入 {self.indexed_chunks} 个 chunk，失败 {len(self.failures)} 个，"
            f"耗时 {self.elapsed_ms} ms"
        )
