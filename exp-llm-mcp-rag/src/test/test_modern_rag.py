from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Sequence

import pytest

from modern_rag.models import INDEX_SCHEMA_VERSION, DocumentChunk, SearchHit
from modern_rag.provenance import SourceLineMapper, extract_source_locator
from modern_rag.providers import BGEReranker
from modern_rag.retrieval import reciprocal_rank_fusion
from modern_rag.service import KnowledgeRAG
from modern_rag.storage import ChromaVectorStore
from rag_agent import RAGAgent, RAG_SYSTEM_GUIDANCE


def test_pdf_provenance_builds_page_range_and_heading_citation(tmp_path: Path) -> None:
    path = tmp_path / "book.pdf"
    meta = {
        "headings": ["第三章", "3.2 重排序"],
        "doc_items": [
            {
                "self_ref": "#/texts/11",
                "label": "text",
                "text": "第一段",
                "prov": [
                    {
                        "page_no": 34,
                        "bbox": {"l": 1, "t": 2, "r": 3, "b": 4},
                        "charspan": [0, 3],
                    }
                ],
            },
            {
                "self_ref": "#/texts/12",
                "label": "text",
                "text": "第二段",
                "prov": [
                    {
                        "page_no": 36,
                        "bbox": {"l": 5, "t": 6, "r": 7, "b": 8},
                        "charspan": [0, 3],
                    }
                ],
            },
        ],
    }

    locator = extract_source_locator(path, meta, 7, "第一段\n第二段")
    metadata = locator.to_metadata(content_hash="hash", index_fingerprint="fp")

    assert locator.location == "第 34–36 页 · 第三章 / 3.2 重排序"
    assert locator.citation == "【book.pdf｜第 34–36 页 · 第三章 / 3.2 重排序】"
    assert metadata["page_start"] == 34
    assert metadata["page_end"] == 36
    provenance = json.loads(str(metadata["provenance"]))
    assert provenance[0]["ref"] == "#/texts/11"
    assert provenance[0]["bbox"]["l"] == 1
    assert str(tmp_path) not in locator.citation


def test_docx_citation_uses_heading_and_element_anchors(tmp_path: Path) -> None:
    path = tmp_path / "contract.docx"
    meta = {
        "headings": ["第四条", "付款安排"],
        "doc_items": [
            {"self_ref": "#/texts/11", "label": "text", "text": "首付款"},
            {"self_ref": "#/texts/12", "label": "text", "text": "里程碑款"},
            {"self_ref": "#/tables/1", "label": "table"},
        ],
    }

    locator = extract_source_locator(path, meta, 3, "付款安排")

    assert locator.location == "第四条 / 付款安排 · 文本项 12–13、表格 2"
    assert locator.citation == "【contract.docx｜第四条 / 付款安排 · 文本项 12–13、表格 2】"


def test_markdown_citation_maps_docling_text_to_source_lines(tmp_path: Path) -> None:
    path = tmp_path / "guide.md"
    path.write_text(
        "# 部署\n\n**付款条件**：货到后三十日内付款。\n补充说明。\n",
        encoding="utf-8",
    )
    mapper = SourceLineMapper.from_path(path)
    assert mapper is not None
    meta = {
        "headings": ["部署"],
        "doc_items": [
            {
                "self_ref": "#/texts/1",
                "label": "text",
                "text": "付款条件：货到后三十日内付款。",
            },
            {
                "self_ref": "#/texts/2",
                "label": "text",
                "text": "补充说明。",
            },
        ],
    }

    locator = extract_source_locator(
        path,
        meta,
        1,
        "付款条件：货到后三十日内付款。\n补充说明。",
        mapper,
    )

    assert locator.line_start == 3
    assert locator.line_end == 4
    assert locator.citation == "【guide.md｜部署 · 第 3–4 行】"


def test_unreliable_line_mapping_falls_back_without_inventing_lines(
    tmp_path: Path,
) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("完全不同的原文", encoding="utf-8")
    mapper = SourceLineMapper.from_path(path)
    assert mapper is not None

    locator = extract_source_locator(
        path,
        {"doc_items": [{"self_ref": "#/texts/4", "text": "无法匹配内容"}]},
        2,
        "无法匹配内容",
        mapper,
    )

    assert locator.line_start is None
    assert locator.location == "结构块 2"
    assert "行" not in locator.citation


def test_rrf_is_deterministic_and_rewards_cross_retriever_hits() -> None:
    fused = reciprocal_rank_fusion(
        [["dense-only", "both"], ["both", "keyword-only"]],
        rank_constant=60,
    )

    assert fused[0][0] == "both"
    assert {item_id for item_id, _ in fused} == {
        "dense-only",
        "both",
        "keyword-only",
    }


def test_bge_reranker_applies_relevance_threshold() -> None:
    class FakeModel:
        def compute_score(self, pairs, normalize=True):
            assert normalize is True
            return [0.91, 0.04]

    reranker = BGEReranker("fake", use_fp16=False, min_score=0.1)
    reranker._model = FakeModel()
    hits = [
        SearchHit(id="a", text="相关", metadata={"source": "a"}),
        SearchHit(id="b", text="无关", metadata={"source": "b"}),
    ]

    result = reranker.rerank("问题", hits, 5)

    assert [hit.id for hit in result] == ["a"]
    assert result[0].rerank_score == pytest.approx(0.91)


def test_chroma_failed_new_version_write_keeps_previous_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ChromaVectorStore(
        persist_dir=tmp_path / "chroma",
        collection_name="rollback_test",
        embedding_model="test-embedding",
        index_schema_version=INDEX_SCHEMA_VERSION,
    )
    source = str((tmp_path / "contract.txt").resolve())

    def chunk(item_id: str, text: str, fingerprint: str, count: int) -> DocumentChunk:
        return DocumentChunk(
            id=item_id,
            text=text,
            embedding_text=text,
            source=source,
            location="第 1 行",
            metadata={
                "source": source,
                "source_name": "contract.txt",
                "location": "第 1 行",
                "citation": "【contract.txt｜第 1 行】",
                "index_fingerprint": fingerprint,
                "source_chunk_count": count,
            },
        )

    old_chunk = chunk("old", "旧版本", "old-fingerprint", 1)
    store.replace_source(source, [old_chunk], [[1.0, 0.0]], batch_size=1)

    original_upsert = store.collection.upsert
    call_count = 0

    def fail_second_batch(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise RuntimeError("模拟第二批写入失败")
        return original_upsert(*args, **kwargs)

    monkeypatch.setattr(store.collection, "upsert", fail_second_batch)
    new_chunks = [
        chunk("new-1", "新版本一", "new-fingerprint", 2),
        chunk("new-2", "新版本二", "new-fingerprint", 2),
    ]

    with pytest.raises(RuntimeError, match="第二批"):
        store.replace_source(
            source,
            new_chunks,
            [[0.9, 0.1], [0.8, 0.2]],
            batch_size=1,
        )

    assert store.source_fingerprint(source) == "old-fingerprint"
    assert [hit.id for hit in store.get_all()] == ["old"]


class FakeSettings:
    embedding_batch_size = 2
    dense_k = 5
    keyword_k = 5
    fusion_k = 5
    final_k = 3
    rrf_constant = 60
    jieba_user_dict = None
    prune_missing_sources = True

    @staticmethod
    def index_fingerprint(content_hash: str) -> str:
        return f"profile:{content_hash}"


class FakeIngestor:
    def __init__(self) -> None:
        self.calls = 0

    def discover(self, target: Path) -> list[Path]:
        if target.is_file():
            return [target]
        return sorted(target.glob("*.txt"))

    def ingest_file(
        self,
        path: Path,
        *,
        content_hash: str,
        index_fingerprint: str,
    ) -> list[DocumentChunk]:
        self.calls += 1
        source = str(path.resolve())
        text = path.read_text(encoding="utf-8")
        metadata = {
            "source": source,
            "source_name": path.name,
            "location": "第 1 行",
            "citation": f"【{path.name}｜第 1 行】",
            "index_fingerprint": index_fingerprint,
            "source_chunk_count": 1,
        }
        return [
            DocumentChunk(
                id=index_fingerprint,
                text=text,
                embedding_text=f"标题\n{text}",
                source=source,
                location="第 1 行",
                metadata=metadata,
            )
        ]


class FakeEmbeddings:
    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls += 1
        return [[1.0, float(index)] for index, _ in enumerate(texts)]


class FakeReranker:
    def rerank(
        self,
        query: str,
        hits: Sequence[SearchHit],
        limit: int,
    ) -> list[SearchHit]:
        return list(hits[:limit])


class FakeStore:
    def __init__(self) -> None:
        self.records: dict[str, list[SearchHit]] = {}
        self.fail_replace = False

    def count(self) -> int:
        return sum(len(value) for value in self.records.values())

    def source_fingerprint(self, source: str) -> str | None:
        hits = self.records.get(source, [])
        if not hits:
            return None
        return str(hits[0].metadata["index_fingerprint"])

    def list_sources(self) -> set[str]:
        return set(self.records)

    def replace_source(
        self,
        source: str,
        chunks: Sequence[DocumentChunk],
        embeddings: Sequence[Sequence[float]],
        batch_size: int,
    ) -> None:
        assert len(chunks) == len(embeddings)
        assert batch_size == 2
        if self.fail_replace:
            raise RuntimeError("模拟写入失败")
        self.records[source] = [
            SearchHit(id=chunk.id, text=chunk.text, metadata=chunk.metadata)
            for chunk in chunks
        ]

    def delete_sources(self, sources: Sequence[str]) -> None:
        for source in sources:
            self.records.pop(source, None)

    def query_dense(
        self,
        query_embedding: Sequence[float],
        limit: int,
    ) -> list[SearchHit]:
        del query_embedding
        hits = self.get_all()[:limit]
        for rank, hit in enumerate(hits, start=1):
            hit.dense_rank = rank
            hit.dense_distance = 0.1
        return hits

    def get_all(self) -> list[SearchHit]:
        return [hit for values in self.records.values() for hit in values]

    def get_by_ids(self, ids: Sequence[str]) -> list[SearchHit]:
        by_id = {hit.id: hit for hit in self.get_all()}
        return [by_id[item_id] for item_id in ids if item_id in by_id]


def test_index_is_incremental_refreshes_retriever_and_prunes_sources(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("合同付款条件", encoding="utf-8")
    second.write_text("交付条件", encoding="utf-8")
    ingestor = FakeIngestor()
    embeddings = FakeEmbeddings()
    store = FakeStore()
    rag = KnowledgeRAG(
        FakeSettings(),
        ingestor,
        embeddings,
        FakeReranker(),
        store,
    )

    initial = rag.index(tmp_path)
    initial_retriever = rag._retriever
    unchanged = rag.index(tmp_path)
    assert initial.indexed_files == 2
    assert unchanged.skipped_files == 2
    assert ingestor.calls == 2
    assert embeddings.calls == 2
    assert rag._retriever is initial_retriever

    first.write_text("合同付款条件已变更", encoding="utf-8")
    changed = rag.index(tmp_path)
    assert changed.indexed_files == 1
    assert changed.skipped_files == 1
    assert rag._retriever is not initial_retriever

    second_source = str(second.resolve())
    second.unlink()
    pruned = rag.index(tmp_path)
    assert pruned.removed_sources == 1
    assert second_source not in store.records

    first.unlink()
    emptied = rag.index(tmp_path)
    assert emptied.discovered_files == 0
    assert emptied.removed_sources == 1
    assert not store.records


def test_failed_reindex_keeps_last_successful_source(tmp_path: Path) -> None:
    path = tmp_path / "contract.txt"
    path.write_text("旧版本", encoding="utf-8")
    ingestor = FakeIngestor()
    embeddings = FakeEmbeddings()
    store = FakeStore()
    rag = KnowledgeRAG(
        FakeSettings(),
        ingestor,
        embeddings,
        FakeReranker(),
        store,
    )
    first = rag.index(path)
    source = str(path.resolve())
    previous_fingerprint = store.source_fingerprint(source)
    assert first.ok

    path.write_text("新版本", encoding="utf-8")
    store.fail_replace = True
    failed = rag.index(path)

    assert len(failed.failures) == 1
    assert store.source_fingerprint(source) == previous_fingerprint
    assert store.records[source][0].text == "旧版本"


def test_rag_tool_returns_only_grounded_evidence_contract() -> None:
    class FakeRAG:
        def search(self, query: str, limit: int | None = None) -> list[SearchHit]:
            assert query == "付款条件"
            assert limit == 2
            return [
                SearchHit(
                    id="evidence-1",
                    text="首付款为 30%。",
                    metadata={
                        "source": r"D:\\documents\\sample_contract.pdf",
                        "source_name": "sample_contract.pdf",
                        "location": "第 1 页 · 第四条 / 付款安排",
                        "citation": "【sample_contract.pdf｜第 1 页 · 第四条 / 付款安排】",
                    },
                    dense_rank=1,
                    keyword_rank=1,
                    rrf_score=0.03,
                    rerank_score=0.9,
                )
            ]

    agent = RAGAgent(model="test", mcp_clients=[], rag=FakeRAG())  # type: ignore[arg-type]
    result = asyncio.run(agent._search_knowledge_base("付款条件", 2))

    assert result.success
    match = result.data["matches"][0]
    assert match == {
        "evidence_id": "evidence-1",
        "citation": "【sample_contract.pdf｜第 1 页 · 第四条 / 付款安排】",
        "document": "sample_contract.pdf",
        "location": "第 1 页 · 第四条 / 付款安排",
        "content": "首付款为 30%。",
    }
    assert "D:\\documents" not in json.dumps(result.data)
    assert "rrf_score" not in match
    assert "[1]" not in RAG_SYSTEM_GUIDANCE
    assert "不得断言\n知识库一定没有该内容" in RAG_SYSTEM_GUIDANCE


def test_rag_tool_marks_empty_retrieval_as_insufficient_evidence() -> None:
    class EmptyRAG:
        def search(self, query: str, limit: int | None = None) -> list[SearchHit]:
            return []

    agent = RAGAgent(model="test", mcp_clients=[], rag=EmptyRAG())  # type: ignore[arg-type]
    result = asyncio.run(agent._search_knowledge_base("不存在的事实", 3))

    assert result.success
    assert result.data == {
        "query": "不存在的事实",
        "evidence_count": 0,
        "insufficient_evidence": True,
        "matches": [],
    }
