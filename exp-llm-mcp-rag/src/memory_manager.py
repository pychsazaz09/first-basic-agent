"""
memory_manager.py — Agent 记忆管理器（Day 14）

短期记忆：Redis 存对话摘要（TTL 自动过期）
长期记忆：ChromaDB 存历史问答（语义检索）
无降级——Redis/Chroma 不可用时抛异常，由上层 Agent 处理。
"""

import os
from typing import Optional


# ═══════════════════════════════════════════════════════════════
# Redis 短期记忆（懒加载，连不上就抛异常）
# ═══════════════════════════════════════════════════════════════
_redis = None

def _get_redis():
    global _redis
    if _redis is not None:
        return _redis
    from redis.asyncio import Redis
    _redis = Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", "6379")),
        decode_responses=True,
    )
    return _redis


# ═══════════════════════════════════════════════════════════════
# Chroma 长期记忆（懒加载，连不上就抛异常）
# ═══════════════════════════════════════════════════════════════
_chroma_collection = None

def _get_chroma():
    global _chroma_collection
    if _chroma_collection is not None:
        return _chroma_collection
    import chromadb
    client = chromadb.PersistentClient("./chroma_memory")
    # metadata 索引：声明 user_id 字段需要建索引，where 过滤才能走索引而非全扫描
    _chroma_collection = client.get_or_create_collection(
        "agent_conversations",
        metadata={"hnsw:space": "cosine"},  # 向量相似度用余弦距离（中文语义更准）
    )
    return _chroma_collection


# ═══════════════════════════════════════════════════════════════
# MemoryManager
# ═══════════════════════════════════════════════════════════════
class MemoryManager:

    def __init__(self, ttl: int = 1800):
        self.ttl = ttl

    # ── 短期记忆 ──

    async def load_short_term(self, user_id: str) -> Optional[str]:
        """
        加载上次对话摘要。
        有摘要 → 返回字符串。
        无摘要（首次对话）→ 返回 None。
        Redis 操作失败 → 抛异常。
        """
        r = _get_redis()
        result: str | None = await r.get(f"agent:session:{user_id}:summary")  # type: ignore[assignment]
        return result if result else None

    async def save_short_term(self, user_id: str, summary: str) -> None:
        """保存对话摘要，TTL 后自动过期。失败抛异常。"""
        r = _get_redis()
        await r.setex(
            f"agent:session:{user_id}:summary",
            self.ttl,
            summary,
        )

    # ── 长期记忆 ──

    async def search_long_term(self, user_id: str, query: str, k: int = 3) -> list[str]:
        """
        语义检索当前用户最相关的历史对话。
        Chroma 靠 metadata 里的 user_id 过滤——只搜用户自己的数据。
        """
        coll = _get_chroma()
        from openai import AsyncOpenAI
        emb = AsyncOpenAI(
            api_key=os.getenv("EMBEDDING_KEY"),
            base_url=os.getenv("EMBEDDING_BASE_URL"),
        )
        resp = await emb.embeddings.create(
            model="BAAI/bge-large-zh-v1.5",
            input=query,
        )
        results = coll.query(
            query_embeddings=[resp.data[0].embedding],
            n_results=k,
            where={"user_id": user_id},       # ← 只搜当前用户的记录
            include=["documents"],
        )
        if results and results.get("documents"):
            return results["documents"][0]  # type: ignore
        return []

    async def save_long_term(
        self, user_id: str, question: str, answer: str, metadata: dict | None = None
    ) -> None:
        """把一轮问答存入长期记忆。user_id 写入 metadata，检索时用 where 过滤。"""
        import uuid
        document = f"问题: {question}\n回答: {answer}"

        coll = _get_chroma()
        from openai import AsyncOpenAI
        emb = AsyncOpenAI(
            api_key=os.getenv("EMBEDDING_KEY"),
            base_url=os.getenv("EMBEDDING_BASE_URL"),
        )
        resp = await emb.embeddings.create(
            model="BAAI/bge-large-zh-v1.5",
            input=document,
        )
        # 把 user_id 合并进 metadata，Chroma 查询时用 where 过滤
        meta = metadata or {}
        meta["user_id"] = user_id
        coll.add(
            embeddings=[resp.data[0].embedding],
            ids=[uuid.uuid4().hex[:12]],
            documents=[document],
            metadatas=[meta],
        )
