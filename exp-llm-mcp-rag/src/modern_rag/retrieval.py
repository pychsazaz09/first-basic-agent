from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from .contracts import EmbeddingProvider, Reranker, VectorStore
from .models import SearchHit


def chinese_tokens(text: str) -> list[str]:
    try:
        import jieba
    except ImportError as exc:
        raise RuntimeError("缺少 jieba") from exc

    tokens = []
    for token in jieba.lcut_for_search(text.lower()):
        token = token.strip()
        if token and re.search(r"[\w一-鿿]", token):
            tokens.append(token)
    return tokens


def _keyword_text(hit: SearchHit) -> str:
    raw_headings = hit.metadata.get("heading_path")
    if not isinstance(raw_headings, str):
        return hit.text
    try:
        headings = json.loads(raw_headings)
    except json.JSONDecodeError:
        headings = []
    if not isinstance(headings, list):
        return hit.text
    heading_text = "\n".join(str(item) for item in headings if item)
    return f"{heading_text}\n{hit.text}" if heading_text else hit.text


class KeywordIndex:
    """In-memory BM25 snapshot refreshed only after index mutations."""

    def __init__(
        self,
        records: Sequence[SearchHit],
        user_dict: Path | None = None,
    ) -> None:
        try:
            import jieba
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise RuntimeError("缺少 jieba 或 rank-bm25") from exc

        if user_dict:
            if not user_dict.exists():
                raise FileNotFoundError(f"jieba 用户词典不存在：{user_dict}")
            jieba.load_userdict(str(user_dict))

        self.records = list(records)
        corpus = [
            chinese_tokens(_keyword_text(record)) or ["__empty__"]
            for record in records
        ]
        self.model = BM25Okapi(corpus) if corpus else None

    def search(self, query: str, limit: int) -> list[SearchHit]:
        query_tokens = chinese_tokens(query)
        if not self.model or not query_tokens:
            return []
        scores = self.model.get_scores(query_tokens)
        order = sorted(
            range(len(scores)),
            key=lambda index: (-float(scores[index]), self.records[index].id),
        )

        hits: list[SearchHit] = []
        for index in order:
            score = float(scores[index])
            if score <= 0:
                continue
            original = self.records[index]
            hits.append(
                SearchHit(
                    id=original.id,
                    text=original.text,
                    metadata=original.metadata,
                    keyword_rank=len(hits) + 1,
                )
            )
            if len(hits) >= limit:
                break
        return hits


def reciprocal_rank_fusion(
    rank_lists: Sequence[Sequence[str]],
    rank_constant: int = 60,
) -> list[tuple[str, float]]:
    fused: defaultdict[str, float] = defaultdict(float)
    for ranking in rank_lists:
        for rank, item_id in enumerate(ranking, start=1):
            fused[item_id] += 1.0 / (rank_constant + rank)
    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))


class HybridRetriever:
    def __init__(
        self,
        store: VectorStore,
        embeddings: EmbeddingProvider,
        reranker: Reranker,
        dense_k: int,
        keyword_k: int,
        fusion_k: int,
        final_k: int,
        rank_constant: int,
        jieba_user_dict: Path | None,
    ) -> None:
        self.store = store
        self.embeddings = embeddings
        self.reranker = reranker
        self.dense_k = dense_k
        self.keyword_k = keyword_k
        self.fusion_k = fusion_k
        self.final_k = final_k
        self.rank_constant = rank_constant
        self.keyword_index = KeywordIndex(store.get_all(), jieba_user_dict)

    def retrieve(self, query: str, limit: int | None = None) -> list[SearchHit]:
        final_k = limit or self.final_k
        query_embedding = self.embeddings.embed([query])[0]
        dense_hits = self.store.query_dense(
            query_embedding,
            max(self.dense_k, final_k),
        )
        keyword_hits = self.keyword_index.search(
            query,
            max(self.keyword_k, final_k),
        )

        rank_lists: list[list[str]] = [[hit.id for hit in dense_hits]]
        if keyword_hits:
            rank_lists.append([hit.id for hit in keyword_hits])
        fused = reciprocal_rank_fusion(rank_lists, self.rank_constant)
        fused = fused[: max(self.fusion_k, final_k)]

        dense_by_id = {hit.id: hit for hit in dense_hits}
        dense_ranks = {hit.id: rank for rank, hit in enumerate(dense_hits, start=1)}
        keyword_ranks = {
            hit.id: rank for rank, hit in enumerate(keyword_hits, start=1)
        }
        fused_scores = dict(fused)
        fused_ids = [item_id for item_id, _ in fused]
        candidate_map = {hit.id: hit for hit in self.store.get_by_ids(fused_ids)}

        ordered: list[SearchHit] = []
        seen: set[tuple[str, str]] = set()
        for item_id in fused_ids:
            hit = candidate_map.get(item_id)
            if not hit:
                continue
            evidence_key = (hit.source, " ".join(hit.text.split()))
            if evidence_key in seen:
                continue
            seen.add(evidence_key)
            dense_hit = dense_by_id.get(item_id)
            hit.dense_rank = dense_ranks.get(item_id)
            hit.dense_distance = (
                dense_hit.dense_distance if dense_hit is not None else None
            )
            hit.keyword_rank = keyword_ranks.get(item_id)
            hit.rrf_score = fused_scores[item_id]
            ordered.append(hit)

        return self.reranker.rerank(query, ordered, final_k)
