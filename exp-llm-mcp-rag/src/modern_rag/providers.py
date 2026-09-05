from __future__ import annotations

from collections.abc import Sequence

from .models import SearchHit


def _openai_client(api_key: str, base_url: str | None):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("缺少 openai Python SDK") from exc

    options: dict[str, object] = {
        "api_key": api_key,
        "timeout": 120.0,
        "max_retries": 3,
    }
    if base_url:
        options["base_url"] = base_url
    return OpenAI(**options)


class OpenAIEmbeddingProvider:
    """Embedding adapter for OpenAI and OpenAI-compatible endpoints."""

    def __init__(self, api_key: str, base_url: str | None, model: str) -> None:
        self.client = _openai_client(api_key, base_url)
        self.model = model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(
            model=self.model,
            input=list(texts),
            encoding_format="float",
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        if len(ordered) != len(texts):
            raise RuntimeError("嵌入接口返回的向量数量与输入文本数量不一致")
        return [list(item.embedding) for item in ordered]


class BGEReranker:
    """Local BGE cross-encoder, loaded only on the first search."""

    def __init__(
        self,
        model_name: str,
        use_fp16: bool,
        min_score: float | None,
    ) -> None:
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.min_score = min_score
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from FlagEmbedding import FlagReranker
            except ImportError as exc:
                raise RuntimeError(
                    "已开启 RAG_ENABLE_RERANKER，但缺少 FlagEmbedding"
                ) from exc
            self._model = FlagReranker(
                self.model_name,
                use_fp16=self.use_fp16,
            )
        return self._model

    def rerank(
        self,
        query: str,
        hits: Sequence[SearchHit],
        limit: int,
    ) -> list[SearchHit]:
        if not hits:
            return []
        pairs = [[query, hit.text] for hit in hits]
        raw_scores = self._get_model().compute_score(pairs, normalize=True)
        if isinstance(raw_scores, (int, float)):
            scores = [float(raw_scores)]
        else:
            scores = [float(score) for score in raw_scores]
        if len(scores) != len(hits):
            raise RuntimeError("reranker 返回的分数数量与候选数量不一致")

        ranked = list(hits)
        for hit, score in zip(ranked, scores):
            hit.rerank_score = score
        ranked.sort(
            key=lambda hit: (
                hit.rerank_score
                if hit.rerank_score is not None
                else float("-inf")
            ),
            reverse=True,
        )
        if self.min_score is not None:
            ranked = [
                hit
                for hit in ranked
                if hit.rerank_score is not None
                and hit.rerank_score >= self.min_score
            ]
        return ranked[:limit]


class PassthroughReranker:
    """Keep RRF order when BGE reranking is disabled."""

    def rerank(
        self,
        query: str,
        hits: Sequence[SearchHit],
        limit: int,
    ) -> list[SearchHit]:
        del query
        return list(hits[:limit])
