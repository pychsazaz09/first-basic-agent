# 2026-07-21 学习日志

> Python 新人（Java 背景）→ RAG + MCP + LLM Agent 项目实战

---

## 一、TS → Python 代码翻译（3 个文件）

| 文件 | 做了什么 |
|------|---------|
| [agent.py](src/agent.py) | Agent 类：LLM ↔ MCP 工具调用循环，`init()` → `invoke()` → `close()`，每行对照 TS 加注释 |
| [VectorStore.py](src/VectorStore.py) | 向量存储：`add_embedding()` + `search()` + `_cosine_similarity()`，纯数组不用 numpy |
| [EmbeddingRetriever.py](src/EmbeddingRetriever.py) | 嵌入检索器：调硅基流动 API（`requests.post`）、存向量、检索，按官方文档写法 |

---


---

## 三、Java → Python 语法问答（共 14 个）

| 你的问题 | 答案 |
|---------|------|
| `@dataclass` 是接口吗？        |  不是，是数据类（= Java 的 record / POJO）。接口用 ABC 或 Protocol |
| Python 没有 `interface` 怎么办？ | ABC（抽象基类）、Protocol（鸭子类型）、不写（最主流） |
| Python 没有 `private` 怎么办？ | `_` 前缀约定（拦不住，靠自觉） |
| `vars(obj)` 是什么？            | 内置函数，返回对象的 `__dict__`（等于把对象转成 dict） |
| `dict.items()` 是什么？       | 同时遍历 key + value，等同 Java 的 `Map.entrySet()` |
| Python 有 `int` 吗？          | `int`（无限大）、`float`（64位）、`bool`（True/False），没有 long/double/short |
| 为什么 `_embed` 不用 `async`？ | `requests` 是同步库，不能 `await` |
| `embed_query` 为什么标 `async`？ | 目前没必要，留着是为了以后换 `httpx` 不用改调用方 |
| `data["data"][0]["embedding"]` 怎么理解？ | `data` → dict 字段 → 数组[0] → 第一个结果的 embedding 字段 |
| 为什么是数组 + 取第一个？       | API 支持一次传多个文本，每个文本对应数组一个元素 |
| `.map()` 怎么转 Python？      | `[f(x) for x in list]` 列表推导式 |
| `() => x.score` 怎么转？        | `lambda x: x["score"]` |
| `.slice(0, n)` 怎么转？        | `list[:n]` 切片 |
| `.reduce((s,a)=>s+a*b[idx], 0)` 怎么转？ | `sum(a*b for a,b in zip(A,B))` |

---

## 四、架构/设计理解（共 9 个）

| 你的问题 | 核心答案 |
|---------|---------|
| 为什么 LLM 不跟 MCP 一起在 `__init__` 创建？ | 两阶段初始化：`__init__` 不能 async，tools 是运行时从 MCP 服务器拿的 |
| `MCPClient` 的 `command` + `args` 是干嘛的？ | 启动 MCP 服务器子进程的命令行（如 `npx -y @modelcontextprotocol/server-filesystem .`） |
| API key 怎么被用到的？ | `test.py` 调 `load_dotenv()` → `os.environ` → `ChatOpenAI` 内部 `os.getenv()` 读取（隐式） |
| MCP filesystem 权限怎么控制？ | 最后一个 `args` 参数指定允许目录，MCP 沙箱拒绝目录外的任何操作 |
| 两个文件设计 I/O 了吗？ | 唯一 I/O：`_embed()` 里的 `requests.post()` 发 HTTP。其他操作（add_embedding、search）都是纯内存 |
| 为什么 `embed_document` 要标 `async`？ | 现在多余——`_embed` 是同步的。留着为了将来换 `httpx` |
| 一篇长文章怎么处理？ | 需要分块（chunking）：模型限制 512 token（~400 字），超出的被截断 |
| `system_prompt` 的作用？ | 软约束——告诉 LLM 它的工作边界，防止越权操作 |
| `.env` 加载在哪里？ | `mcpCient.py` 模块 import 时自动 `load_dotenv()`，`test.py` 也加了一条 |

---

## 五、工具/环境配置（共 5 个）

| 你的问题 | 解决 |
|---------|------|
| 怎么运行 `.py` 文件？ | `python src/test/test.py`（在项目根目录下执行） |
| VS Code 怎么装 Continue？ | `Ctrl+Shift+X` → 搜 `Continue` → Install |
| Continue 怎么配 DeepSeek？ | 修改 `~/.continue/config.yaml`：`model: deepseek-chat`，`provider: deepseek` |
| Continue 怎么用？ | Tab 接受内联建议、`Ctrl+L` 发选中代码到聊天、右侧面板对话 |
| `requests` 没安装怎么解决？ | `pip install requests` |

---

## 六、概念科普（共 5 个）

| 概念 | 一句话 |
|------|--------|
| 余弦相似度 | `(A·B)/(|A|×|B|)`，值越大方向越一致 → 越相似，用来做语义搜索 |
| RAG 流程 | 用户问题→转向量→VectorStore.search()→相关文档+问题→LLM 生成答案 |
| MCP 协议 | 你的程序通过子进程 stdin/stdout 和 MCP 服务器通信（JSON-RPC） |
| 两阶段初始化 | `__init__` 存配置（同步）+ `init()` 连接外部（异步），因为构造函数不能 async |
| GBK vs UTF-8 | Windows 中文终端默认 GBK，遇到中文/特殊字符就崩 → `sys.stdout.reconfigure(encoding="utf-8")` |

---

## 七、代码对照速查表

### TS/JS → Python

```
TS/Java                           Python
═══════════════════════════════   ═════════════════════════════
this                              self（必须显式写在方法参数第一位）
null                              None
new Xxx()                         Xxx()（没有 new 关键字）
interface / record                @dataclass
Map<K, V>                         dict[K, V]
List<E>                           list[E]
new ArrayList<>()                 []
list.push(item)                   list.append(item)
list.map(x => f(x))               [f(x) for x in list]          ← 列表推导式
list.filter(x => cond)            [x for x in list if cond]
list.sort((a,b) => b-a)           list.sort(key=lambda, reverse=True)
list.slice(0, n)                  list[:n]                       ← 切片
array.reduce((s,a)=>s+a, 0)       sum(array)
(x) => x.score                   lambda x: x["score"]
JSON.stringify(obj) / parse(s)    json.dumps(obj) / loads(s)
fetch(url, {method,headers,body}) requests.post(url, headers=, json=)   ← 同步
                                  httpx.AsyncClient().post(...)          ← 异步
process.env.KEY                   os.getenv("KEY")
console.log(x)                    print(x)
Math.sqrt(x)                      math.sqrt(x)（需要 import math）
Object.keys(obj)                  obj.keys()
private                           _ 前缀约定
throw new Error("msg")            raise Exception("msg")
try { ... } catch (e) { }         try: ... except E as e:
async function / await            async def / await（完全一样）
export default class Xxx          class Xxx（Python 没有 export default）
```

### MCP 工具对照表

```
功能                command     args
────────────────    ───────     ──────────────────────────────────────
文件系统 (npm)       npx         -y @modelcontextprotocol/server-filesystem <允许目录>
文件系统 (Python)    uvx         @modelcontextprotocol/server-filesystem <允许目录>
网页抓取 (npm)       npx         -y @anthropic-ai/mcp-server-fetch
网页抓取 (Python)    mcp-server-fetch  （先 uv tool install）
```

