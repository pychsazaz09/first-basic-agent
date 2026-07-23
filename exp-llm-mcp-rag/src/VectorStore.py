"""
VectorStore - 向量存储（RAG 的记忆库）

═══════════════════════════════════════════════════════
Java → Python 速查（先看这个）
═══════════════════════════════════════════════════════
  Java                             Python
  ───────────────────────────      ──────────────────────
  interface / record               @dataclass
  List<T>                          list[T]
  new ArrayList<>()                []  （空列表）
  list.push(item)                  list.append(item)
  list.map(x -> f(x))              [f(x) for x in list]    ← 列表推导式
  list.sort((a,b)->b-a)            list.sort(key=lambda, reverse=True)
  list.subList(0, n)               list[:n]                ← 切片
  Math.sqrt(x)                     math.sqrt(x)
  array.reduce((s,a)=>s+a, 0)      sum(array)
  private                          _   （前缀约定）
═══════════════════════════════════════════════════════
"""

import math                     # Python 的 Math 库（TS 的 Math 是内置的）
from dataclasses import dataclass  # 自动生成 __init__ __repr__ __eq__


# ============================================================
# @dataclass — Python 的数据类
#
# 贴上 @dataclass 装饰器后，Python 自动帮你生成：
#   __init__()   = 构造函数
#   __repr__()   = 打印格式（Java 的 toString()）
#   __eq__()     = 相等比较（Java 的 equals() + hashCode()）
#
# 不用自己写任何 boilerplate，声明字段就行。
#
# TS 等价物：
#   interface VectorStoreItem {
#       embedding: number[];
#       document: string;
#   }
# 但 @dataclass 更像 Java 的 record —— 编译后自动生成方法。
# ============================================================

@dataclass
class VectorStoreItem:
    """
    向量存储里的一条记录。

    每个文档存进来时，会被转成一个向量（embedding），
    和原文（document）一起存到 VectorStoreItem 里。

    字段说明：
      embedding : 文档的嵌入向量（浮点数列表，比如 [0.01, -0.5, 0.3, ...]）
                  向量"代表"了这个文档的语义含义
      document  : 原始文本（搜索命中后要返回给 LLM 看的）
    """
    embedding: list[float]   # TS: number[]。浮点数列表 = 向量
    document: str            # TS: string。原始文档内容


# ============================================================
# VectorStore — 内存向量存储
#
# 本质：一个 list + 余弦相似度 + 排序
#
# 内部结构（对照 TS）：
#   TS:  private vectorStore: VectorStoreItem[]
#   PY:  self._store: list[VectorStoreItem]
#
# _ 前缀 = Python 约定"我是内部属性，别在外面直接访问"
# Python 没有 private 关键字，全靠命名约定和程序员的自觉
# ============================================================

class VectorStore:

    # ----------------------------------------------------------
    # 构造函数
    #
    # TS:  constructor() { this.vectorStore = []; }
    # PY:  def __init__(self) -> None:
    #
    # self = 当前实例（Java 的 this，TS 的 this）
    # 必须写在每个方法的第一个参数位置，调用时不需要传
    #
    # -> None 表示没有返回值（类似 TS 的 void），不写也行
    # ----------------------------------------------------------
    def __init__(self) -> None:
        """初始化空的向量存储"""
        # list[VectorStoreItem] = TS 的 VectorStoreItem[]
        # [] = TS 的 [] = Java 的 new ArrayList<>()
        self._store: list[VectorStoreItem] = []

    # ----------------------------------------------------------
    # 添加文档
    #
    # TS 对应：
    #   async addEmbedding(embedding: number[], document: string) {
    #       this.vectorStore.push({ embedding, document });
    #   }
    #
    # Python 不需要 async —— TS 版本虽然写了 async 但内部没有 await，
    # Python 直接写成普通方法就行。
    #
    # TS 的 .push({ embedding, document }) 是 JS 的对象简写，
    # Python 要写成 VectorStoreItem(embedding=embedding, document=document)
    # ----------------------------------------------------------
    def add_embedding(self, embedding: list[float], document: str) -> None:
        """
        存入一个文档及其嵌入向量。

        参数：
          embedding : 文档的向量表示（调用 embedding API 得到的）
          document  : 原始文本

        示例：
          store.add_embedding([0.01, -0.5, 0.3], "Python 是动态语言")
        """
        # .append(item) = TS 的 .push(item) = Java 的 list.add(item)
        self._store.append(VectorStoreItem(
            embedding=embedding,    # TS: { embedding, document }
            document=document,      # 同名简写 → 显式写 key=value
        ))

    # ----------------------------------------------------------
    # 语义搜索 —— 核心方法
    #
    # TS 对应：
    #   async search(queryEmbedding, topK = 3): Promise<string[]> {
    #       const scored = this.vectorStore.map((item) => ({
    #           document: item.document,
    #           score: this.cosineSimilarity(queryEmbedding, item.embedding),
    #       }));
    #       const topKDocuments = scored
    #           .sort((a, b) => b.score - a.score)
    #           .slice(0, topK)
    #           .map((item) => item.document);
    #       return topKDocuments;
    #   }
    #
    # Python 对照（逐一写注释）：
    # ----------------------------------------------------------
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 3,          # TS: topK = 3，默认返回前 3 个最相似的
    ) -> list[str]:
        """
        找跟查询向量最相似的 top_k 篇文档。

        流程（和 TS 完全一样）：
          1. 遍历每条记录，算余弦相似度（得到一个分数）
          2. 按分数降序排列（分高的在前）
          3. 取前 top_k 个
          4. 只返回 document 文本（丢掉分数）

        参数：
          query_embedding : 用户问题被 embed 后的向量
          top_k           : 返回前几名，默认 3

        返回：
          list[str] —— 最相似的文档内容列表（纯文本）
        """

        # ---------- 第 1 步：算分 ----------
        # TS: const scored = this.vectorStore.map((item) => ({...}))
        #
        # Python 用"列表推导式"代替 .map()：
        #   [表达式 for 变量 in 可迭代对象]
        #   等价于 TS: array.map(变量 => 表达式)
        #
        # 每次循环创建一个临时 dict {"document": ..., "score": ...}，
        # 和 TS 里的匿名对象 { document, score } 一样。
        scored = [
            {
                "document": item.document,
                "score": self._cosine_similarity(
                    query_embedding,     # 用户的查询向量
                    item.embedding       # 库里的文档向量
                ),
            }
            for item in self._store      # ← 遍历每条记录
        ]

        # ---------- 第 2 步：排序 ----------
        # TS: scored.sort((a, b) => b.score - a.score)
        #
        # Python 的 .sort() 用 key 参数指定"按啥排序"：
        #   key=lambda x: x["score"]   ← 按 score 字段排
        #   reverse=True               ← 降序（大的在前）
        #
        # lambda x: x["score"] 是 Python 的匿名函数（箭头函数）：
        #   TS:  (x) => x.score
        #   PY:  lambda x: x["score"]
        #   Java: (x) -> x.getScore()
        scored.sort(key=lambda x: x["score"], reverse=True)

        # ---------- 第 3+4 步：切片 + 提取 document ----------
        # TS: .slice(0, topK).map((item) => item.document)
        #
        # list[:top_k] 是 Python 的切片（slice）语法：
        #   list[start:end]  →  从 start 取到 end（不含 end）
        #   list[:n]         →  从开头取到第 n-1 个（共 n 个）
        #
        # 然后套一层列表推导式只取 document 字段：
        #   [item["document"] for item in scored[:top_k]]
        #   ↑ 结果列表               ↑ 遍历切片            ↑ 取 document
        return [item["document"] for item in scored[:top_k]]

    # ----------------------------------------------------------
    # 余弦相似度（内部方法）
    #
    # TS 对应：
    #   private cosineSimilarity(vecA: number[], vecB: number[]): number {
    #       const dotProduct = vecA.reduce((sum, a, idx) => sum + a * vecB[idx], 0);
    #       const normA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
    #       const normB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
    #       return dotProduct / (normA * normB);
    #   }
    #
    # Python 没有 private，_ 前缀 = 约定"别在外面调"
    #
    # 公式：cos(θ) = (A · B) / (|A| × |B|)
    #
    #         分子：点积 = a1×b1 + a2×b2 + a3×b3 + ...
    #         分母：L2 范数的乘积 = sqrt(a1²+a2²+...) × sqrt(b1²+b2²+...)
    #
    # 结果在 -1 到 1 之间：
    #   +1  = 方向完全一致（最相似）
    #    0  = 方向垂直（不相关）
    #   -1  = 方向相反（完全对立）
    # ----------------------------------------------------------
    def _cosine_similarity(
        self,
        vec_a: list[float],
        vec_b: list[float],
    ) -> float:
        """
        计算两个向量的余弦相似度。

        通俗理解：
          把两个向量想象成从原点射出的两根箭。
          箭头方向越一致 → 夹角越小 → cos 越接近 1 → 它们越"相似"。
          "语义搜索"就是：把用户问题转成箭，找库里方向最接近的文档箭。
        """

        # ----- 点积（Dot Product）-----
        # TS: vecA.reduce((sum, a, idx) => sum + a * vecB[idx], 0)
        #
        # zip(A, B) = 把两个列表按位置配对：
        #   A = [1, 2, 3]
        #   B = [4, 5, 6]
        #   zip(A, B) → 生成 (1,4), (2,5), (3,6) 这样的元组
        #   sum(a * b for a, b in zip(A, B)) → 1×4 + 2×5 + 3×6 = 32
        #
        # sum(...) = Python 内置函数，求和
        # a * b for a, b in zip(...) = 生成器表达式（惰性求值的列表推导式）
        dot = sum(a * b for a, b in zip(vec_a, vec_b))

        # ----- L2 范数（向量的欧几里得长度）-----
        # TS: Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0))
        #
        # sum(a * a for a in vec_a) = 每个元素平方再求和（不借助 numpy 的做法）
        # math.sqrt(...) = 开平方（Python 的 Math 需要 import，TS 是全局的）
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))

        # ----- 防除零 -----
        # 向量长度理论上不可能为 0（模型不会生成全 0 向量），但防一下
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        # ----- 余弦相似度 -----
        # dot / (normA * normB)  →  -1 到 1 之间的值
        return dot / (norm_a * norm_b)
