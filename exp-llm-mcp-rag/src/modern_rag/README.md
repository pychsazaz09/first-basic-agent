# RAGAgent 的生产型 RAG 基线

这套实现面向企业内部单租户知识库：保留原文证据和 Docling provenance，离线增量索引，在线执行 Dense + BM25 + RRF + BGE，并把可回到原文验证的 citation 交给现有 Agent。索引入口不暴露给 LLM，模型不能自行扫描目录。

## 方法

### 离线索引

```text
文件发现与内容哈希
  → Docling 解析（OCR、表格、结构）
  → HybridChunker 按结构和 token 切块
  ├─ 原始 chunk text：回答和审计
  └─ 标题/上下文增强文本：embedding
  → 批量 embedding
  → Chroma cosine collection
  → 新版本写完后再清理旧版本
```

每个源文件都有 `index_fingerprint`，由以下内容共同决定：

- 文件内容 SHA-256；
- embedding 模型；
- chunk token 上限；
- OCR 模式和严格转换策略；
- 索引 schema 版本。

fingerprint 相同则跳过解析和 embedding。某个文件解析或 embedding 失败时，该文件上一次成功版本仍保留；目录同步还会清理目录中已经删除的源文件。

### 在线检索

```text
用户问题
  ├─ Dense embedding 检索
  └─ jieba + BM25 中文关键词检索
       ↓
      RRF 融合（不混合不可比的原始分数）
       ↓
      BGE cross-encoder 重排与相关性门槛
       ↓
      去重后的原文证据 + citation
       ↓
      现有 Agent LLM 生成答案
```

BM25 索引是 Chroma 内容的内存快照，只在进程首次检索或索引成功变化后构建，不会每次查询扫描全库。直接调用 `search_knowledge()` 仍可检查 dense rank/distance、BM25 rank、RRF 和 rerank score；这些诊断字段不会发送给生成模型。

## 证据和引用

Docling `DocMeta` 中使用的字段包括：

- `origin`：来源文档；
- `headings`：标题层级；
- `doc_items[].self_ref`：文档元素引用；
- `doc_items[].label`：文本、表格、图片等元素类型；
- `doc_items[].prov[].page_no`：页码；
- `doc_items[].prov[].bbox`：页面坐标；
- `doc_items[].prov[].charspan`：元素内字符范围。

这些嵌套数据会扁平化成 Chroma 支持的字符串、数字和布尔元数据。完整的紧凑 provenance JSON 仍保留用于审计，面向用户的 citation 按格式生成：

```text
PDF   【book.pdf｜第 34–36 页 · 第三章 / 3.2 重排序】
DOCX  【contract.docx｜第四条 / 付款安排 · 文本项 12–16、表格 2】
MD    【guide.md｜部署 / Docker · 第 42–58 行】
TXT   【notes.txt｜第 18–31 行】
```

TXT/MD 行号通过 Docling 元素文本按文档顺序映射回原文件。只有匹配可信时才输出行号；匹配失败时退回标题、文档元素或结构块，不伪造位置。PDF 使用页码而不使用不稳定的“行号”；DOCX 使用标题与元素锚点，因为 Word 页码受字体、打印机和排版环境影响。

工具返回给 LLM 的单条证据结构为：

```json
{
  "evidence_id": "稳定的 chunk ID",
  "citation": "【sample_contract.pdf｜第 1 页 · 第四条 / 付款安排】",
  "document": "sample_contract.pdf",
  "location": "第 1 页 · 第四条 / 付款安排",
  "content": "原始证据文本"
}
```

系统提示要求每个基于知识库的实质性结论原样复制 `citation`。文档内容按不可信数据处理，不能作为指令；检索为空时必须说明证据不足。

## 模块结构

| 文件 | 职责 |
|---|---|
| `models.py` | chunk、hit、source locator、index report 数据契约 |
| `provenance.py` | Docling provenance 提取、文本行映射、citation 格式化 |
| `ingestion.py` | 文件发现、哈希、Docling 解析、HybridChunker |
| `providers.py` | OpenAI 兼容 embedding、BGE reranker |
| `storage.py` | Chroma profile 校验、版本安全替换、查询 |
| `retrieval.py` | 中文 BM25、Dense、RRF、去重、rerank |
| `service.py` | 增量索引、目录同步、检索器快照和并发边界 |
| `factory.py` | 具体实现组装 |
| `../rag_agent.py` | 注册只读知识检索工具并约束生成答案 |

核心组件通过 `contracts.py` 的 Protocol 解耦。数据量或并发超过单节点能力时，可把 `VectorStore` 换成远程向量库，把 BM25 换成 OpenSearch，而摄取、证据契约和 Agent 工具协议不需要重写。

## 配置

```dotenv
EMBEDDING_KEY=你的-key
EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5

RAG_CHROMA_DIR=./rag_chroma
RAG_CHROMA_COLLECTION=agent_knowledge_v2
RAG_CHUNK_MAX_TOKENS=700
RAG_EMBEDDING_BATCH_SIZE=64
RAG_DENSE_K=30
RAG_KEYWORD_K=30
RAG_FUSION_K=30
RAG_FINAL_K=6
RAG_RRF_CONSTANT=60

RAG_PDF_OCR_MODE=default
RAG_STRICT_CONVERSION=true
RAG_PRUNE_MISSING_SOURCES=true

RAG_ENABLE_RERANKER=true
RAG_RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RAG_RERANKER_USE_FP16=false
RAG_RERANKER_MIN_SCORE=0.1
```

`RAG_RERANKER_MIN_SCORE=off` 可关闭相关性门槛。阈值必须用企业自己的问答金标集调优，`0.1` 只是当前 BGE normalized score 的保守起点。

## 使用

```python
report = await agent.index_documents(r"D:\documents")
print(report)
for failure in report.failures:
    print(failure.source, failure.error)

hits = await agent.search_knowledge("合同付款条件是什么？")
for hit in hits:
    print(hit.citation, hit.rerank_score)

answer = await agent.invoke("根据知识库说明合同付款条件，并引用来源")
```

索引 schema 已升级，默认 collection 是 `agent_knowledge_v2`。旧 collection 不会静默混入新 embedding 空间；首次使用必须重新运行 `index_documents()`。

## 生产边界

当前实现是可部署的单节点基线，适合一本长书到中等规模内部文档库。真正的多租户企业服务还必须在应用入口提供用户身份，并在检索前强制执行 tenant/ACL 过滤；不能依赖提示词做权限控制。更大规模应使用任务队列处理摄取、对象存储保存原文、OpenSearch 提供分布式 sparse 检索、远程向量数据库提供副本与备份，并以领域金标集持续监控 Recall@K、MRR、引用正确率和无答案拒答率。
