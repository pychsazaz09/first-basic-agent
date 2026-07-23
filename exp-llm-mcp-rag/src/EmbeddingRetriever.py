"""
EmbeddingRetriever - 嵌入检索器（RAG 的"翻译官"）

═══════════════════════════════════════════════════════
RAG 里的位置
═══════════════════════════════════════════════════════

  用户提问 → EmbeddingRetriever.embed_query()（提问转成向量）
                ↓
             VectorStore.search()          （找相似的文档）
                ↓
             相关文档 → LLM                 （生成回答）

═══════════════════════════════════════════════════════
职责
═══════════════════════════════════════════════════════
  1. 调硅基流动 embedding API（按官方文档写法）
  2. 把向量 + 原文存到 VectorStore
  3. 接收问题 → 检索相关文档

官方文档：
  https://api-docs.siliconflow.cn/docs/api/embeddings-post
═══════════════════════════════════════════════════════
"""

import os
import requests  # ← 按硅基流动官方文档，用 requests 而不是 httpx
from utils import logTile
from VectorStore import VectorStore


class EmbeddingRetriever:
    """
    嵌入检索器 —— 调 embedding API 把文本转成向量，支持存入和检索。

    TS 对应：
      export default class EmbeddingRetriever {
          private embeddingModel: string;
          private vectorStore: VectorStore;
          async embedDocument(document) { ... }
          async embedQuery(query) { ... }
          private async embed(document): Promise<number[]> { ... }
          async retrieve(query, topK = 3): Promise<string[]> { ... }
      }
    """

    # ---- 构造函数 ----
    # TS: constructor(embeddingModel: string) {
    #         this.embeddingModel = embeddingModel;
    #         this.vectorStore = new VectorStore();
    #     }
    def __init__(self, embedding_model: str="BAAI/bge-large-zh-v1.5") -> None:
        """
        参数：
          embedding_model : 模型名，如 "BAAI/bge-large-zh-v1.5"
        """
        self.embedding_model = embedding_model      # TS: this.embeddingModel
        self.vector_store = VectorStore()            # TS: new VectorStore()

        # 从 .env 读配置（TS: process.env.XXX）
        self.api_key = os.getenv("EMBEDDING_KEY")
        self.base_url = os.getenv("EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1")

    # ---- 嵌入文档 → 转向量 + 存库 ----
    # TS: async embedDocument(document: string)
    async def embed_document(self, document: str) -> list[float]:
        """把文档转成向量并存入 VectorStore"""
        logTile("EMBEDDING DOCUMENT")
        embedding = self._embed(document)            # TS: await this.embed(document)
        self.vector_store.add_embedding(embedding, document)
        return embedding

    # ---- 嵌入查询 → 只转向量，不存库 ----
    # TS: async embedQuery(query: string)
    async def embed_query(self, query: str) -> list[float]:
        """把用户问题转成向量（不存库）"""
        logTile("EMBEDDING QUERY")
        return self._embed(query)                    # TS: await this.embed(query)

    # ---- 一站式检索 ----
    # TS: async retrieve(query, topK = 3): Promise<string[]>
    async def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        """输入问题 → 返回最相关的文档列表"""
        query_embedding = await self.embed_query(query)
        return self.vector_store.search(query_embedding, top_k)

    # ============================================================
    # 核心：调硅基流动 API（按官方文档写法）
    #
    # 官方 Python 示例：
    #   import requests
    #   response = requests.post(
    #       "https://api.siliconflow.cn/v1/embeddings",
    #       headers={
    #           "Authorization": "Bearer $SILICONFLOW_API_KEY",
    #           "Content-Type": "application/json"
    #       },
    #       json={
    #           "input": "Hello, world!",
    #           "model": "Qwen/Qwen3-VL-Embedding-8B"
    #       }
    #   )
    #   print(response.json())
    #
    # TS 版本用 fetch()，Python 用 requests.post()——效果一样。
    # requests 是同步的（阻塞），但因为 _embed 是内部方法，
    # 外部调 embed_document / embed_query 是 async，不影响对外接口。
    # ============================================================

    # TS: private async embed(document): Promise<number[]>
    def _embed(self, document: str) -> list[float]:
        """
        调 embedding API，文本 → 向量。

        参数：document = 待转换的文本
        返回：list[float] = 嵌入向量（如 1024 维的浮点数数组）
        """

        # ---- 前置检查 ----
        if not self.api_key:
            raise RuntimeError(
                "EMBEDDING_KEY 没配！请在 .env 里设置 EMBEDDING_KEY=你的硅基流动key"
            )

        # ---- 发请求（跟官方文档一模一样）----
        # TS: const response = await fetch(baseURL + '/embeddings', {
        #         method: 'POST',
        #         headers: { 'Authorization': `Bearer ${KEY}`, ... },
        #         body: JSON.stringify({ model, input, encoding_format: 'float' }),
        #     })
        #
        # requests.post(url, headers={...}, json={...})
        #   json= 参数自动做 Content-Type + JSON 序列化，不需要手动 json.dumps()
        response = requests.post(
            f"{self.base_url}/embeddings",       # TS: `${baseURL}/embeddings`
            headers={
                "Authorization": f"Bearer {self.api_key}",  # TS: `Bearer ${KEY}`
                "Content-Type": "application/json",
            },
            json={                               # ← 等价于 TS 的 body: JSON.stringify(...)
                "model": self.embedding_model,   # TS: this.embeddingModel
                "input": document,
                "encoding_format": "float",
            },
        )

        # ---- 解析 ----
        # TS: const data = await response.json()
        # PY: response.json() 完全一样（把 HTTP body 的 JSON 转成 dict）
        data = response.json()

        # 错误处理
        if "data" not in data:
            raise RuntimeError(f"Embedding API 返回错误: {data}")

        # ---- 提取向量 ----
        # TS: console.log(data.data[0].embedding)
        #     return data.data[0].embedding
        #
        # Python 的 dict 用 [] 取值（不像 JS 可以用 data.data）
        # data["data"] = 取 data 那个字段
        # [0]          = 数组第一个元素（单文本只有一个结果）
        # ["embedding"] = 那个结果里的 embedding 字段
        embedding = data["data"][0]["embedding"]

        print(f"向量维度: {len(embedding)}, 前5个值: {embedding[:5]}")
        return embedding
