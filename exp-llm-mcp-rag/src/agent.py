"""
Agent - LLM 智能体，编排 LLM ↔ MCP 工具的交互循环

这是从 TypeScript 版本翻译过来的 Python 实现。
Agent 是整个系统的"大脑"——它让 LLM 可以反复调用 MCP 工具，
直到找到最终答案。

Python 新手（尤其是从 Java 转来的）阅读指南：
  1. 先看 class Agent 的结构图（本注释下方）
  2. 再看 __init__（了解初始化做了什么）
  3. 然后看 invoke()（核心循环逻辑）
  4. 最后看 init() 和 close()（生命周期管理）

═══════════════════════════════════════════════════════════════════
Java → Python 速查表（在阅读代码前先扫一遍）
═══════════════════════════════════════════════════════════════════

┌──────────────────────┬──────────────────────────────────────────┐
│ Java                          │ Python                             │
├──────────────────────┼──────────────────────────────────────────┤
│ this                           │ self（必须显式写在方法参数第一位）│
│ null                           │ None                               │
│ new Xxx()                      │ Xxx()（没有 new 关键字）            │
│ private String name;           │ self.name = ""（不需要预先声明）    │
│ List<String>                   │ list[str]                          │
│ Map<String, Object>            │ dict[str, Any]                     │
│ for (X x : list) { ... }       │ for x in list:（冒号 + 缩进）      │
│ try { ... } catch (E e) { }    │ try: ... except E as e:（缩进）    │
│ throw new Exception("msg")     │ raise Exception("msg")             │
│ .stream().map(x -> ...)        │ [f(x) for x in list]（列表推导式） │
│ .stream().flatMap(...)         │ 嵌套列表推导式，或 itertools.chain │
│ .stream().filter(x -> ...)     │ [x for x in list if cond]          │
│ .stream().findFirst()          │ next((x for x in list if c), None) │
│ @Override                      │ 不需要——方法自动覆盖               │
│ implements Interface           │ 不需要显式声明——鸭子类型           │
│ static void main(String[] a)   │ if __name__ == "__main__":         │
│ import com.foo.Bar;            │ from foo import Bar                │
│ System.out.println("hello")    │ print("hello")                     │
│ // 单行注释                    │ # 单行注释（或用 # 开头）           │
│ /* 多行注释 */                 │ ''' 多行文档字符串 '''             │
└──────────────────────┴──────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════
Agent 类的结构图
═══════════════════════════════════════════════════════════════════

┌──────────────────────────────────────────────────────────────────┐
│                        Agent (智能体)                             │
│                                                                   │
│  持有资源：                                                       │
│    • mcp_clients : list[MCPClient] — 一组 MCP 工具服务器          │
│    • llm         : ChatOpenAI — 大语言模型客户端                  │
│                                                                   │
│  生命周期：                                                       │
│    __init__()  →  init()  →  invoke() 循环  →  close()            │
│                                                                   │
│  invoke() 核心循环（Tool-use Loop）：                              │
│    ┌──────────┐     ┌──────────────┐     ┌──────────────┐         │
│    │ 调用 LLM │ ──→ │ LLM 返回     │ ──→ │ 执行工具调用  │         │
│    │ 发送对话 │     │ tool_calls   │     │ 追加结果     │         │
│    └──────────┘     └──────────────┘     └──────────────┘         │
│          ↑                                        │                │
│          └────────────────────────────────────────┘                │
│                    循环直到 LLM 不再要求调用工具                    │
└──────────────────────────────────────────────────────────────────┘
"""

# 导入项目内的类
# 等价于 TS: import MCPClient from "./MCPClient"
# 等价于 Java: import com.xxx.MCPClient;
import json
import uuid
from mcpCient import MCPClient  # MCP 客户端（每个实例连一个 MCP 服务器）
from ChatOpenAI import ChatOpenAI  # LLM 聊天客户端（封装 OpenAI 兼容 API）
from utils import logTile, safe_print,trace_id  # 日志工具

from token_budget import TokenBudget
# agent.py 顶部，和其他 import 放一起
from memory_manager import MemoryManager
from tool_register import ToolRegister,ToolDef
from executor import MCPExecutor


# ============================================================
# Agent - 智能体类
#
# Python 的 class 语法要点（Java 程序员必读）：
#   1. 没有 private/protected/public —— 全靠 _ 前缀约定
#      self.name       → 相当于 public
#      self._name      → 相当于 protected（"别碰，但我管不了你"）
#      self.__name     → 名称改写（name mangling），不是真正的 private
#   2. 不需要在 class 顶部声明成员变量
#      所有成员变量在 __init__ 里直接用 self.xxx = xxx 创建即可
#   3. 继承语法：class Dog(Animal):  等价于 Java 的 class Dog extends Animal
#   4. 多继承是允许的（但容易踩坑，一般不推荐）
# ============================================================
class Agent:
    """
    LLM 智能体 —— 编排 LLM 与 MCP 工具之间的交互循环。

    职责：
      1. 管理多个 MCP 客户端（连不同的 MCP 服务器）
      2. 创建 ChatOpenAI 实例（绑定 tools）
      3. 执行 tool-use 循环：
         a) 把用户问题发给 LLM
         b) 如果 LLM 要求调用工具 → 执行工具 → 把结果发回 LLM
         c) 重复 step b，直到 LLM 直接返回文字答案
      4. 返回最终回复

    简单类比（Java 程序员熟悉的）：
      这就像一个"带插件的命令行解释器"——
      LLM 是解释器，MCP 工具是插件，
      Agent 是那个把插件注册到解释器、然后驱动整个交互的主循环。
    """

    # ----------------------------------------------------------
    # Python 的构造函数：__init__
    #
    # Java 对比:
    #   public class Agent {
    #       private List<MCPClient> mcpClients;
    #       private ChatOpenAI llm = null;
    #       private String model;
    #       private String systemPrompt;
    #       private String context;
    #
    #       public Agent(String model, List<MCPClient> mcpClients,
    #                    String systemPrompt, String context) {
    #           this.mcpClients = mcpClients;
    #           this.model = model;
    #           this.systemPrompt = systemPrompt;
    #           this.context = context;
    #       }
    #   }
    #
    # Python 中：
    #   • __init__ = 构造函数（双下划线开头的方法叫"魔法方法"）
    #   • self = this（但必须显式写在每个方法的第一个参数位置）
    #   • 不需要预先声明成员变量，直接在 __init__ 里面 self.xxx = xxx 即可
    #   • None = Java 的 null
    #   • 类型注解（: str, : list[MCPClient]）是可选的，但写了能让 IDE 更聪明
    # ----------------------------------------------------------
    def __init__(
        self,
        model: str,                        # 模型名称，如 "deepseek-chat"
        mcp_clients: list[MCPClient],      # MCP 客户端列表
        system_prompt: str = "",           # 系统提示词（可选，默认空字符串）
        context: str = "",                 # 初始上下文（可选，默认空字符串）
    ) -> None:
        """
        构造 Agent 实例。

        ⚠️ Python 默认参数大坑（Java 没有这个问题）：
          默认参数的值只会在函数定义时计算一次！
          所以不能用 system_prompt: str = ""（这倒没问题，因为 str 不可变）
          但不能用 mcp_clients: list[MCPClient] = [] ← 大坑！

          为什么？因为 [] 是可变对象，所有实例会共享同一个 list：
            >>> a = Agent("m1", [])
            >>> b = Agent("m2")  # 这次没传，用了默认值
            >>> a.mcp_clients.append(MCPClient(...))
            >>> b.mcp_clients  # 竟然也有数据了！！！

          正确做法是默认为 None 然后在函数体内判断（参见 ChatOpenAI.py 的做法）。
          这里 mcp_clients 没有默认值，所以不涉及这个问题。
        """

        # ----- 一、保存基础配置 -----
        # Python 没有 private 关键字
        # _ 前缀是约定："我是内部属性，外部别碰"
        # 但实际上 Python 不会阻止任何人访问 self._xxx
        # （这就是所谓的"We're all consenting adults here"文化）
        self.model = model
        self.system_prompt = system_prompt
        self.context = context
        self.mcp_clients = mcp_clients
        self.token_budget=TokenBudget("cl100k_base",100000)

        # ----- 二、LLM 客户端：延迟初始化 -----
        # Java: private ChatOpenAI llm = null;
        # Python: 直接赋值 None
        # llm 要等到 init() 里收集完所有 tools 之后才能创建
        self.llm: ChatOpenAI | None = None
        #        ^^^^^^^^^^^^^^^^^^^^^^^^
        #        Python 3.10+ 的联合类型语法：X | Y 等价于 Union[X, Y]
        #        等价于 Java: @Nullable ChatOpenAI llm;
        #        等价于 TS: ChatOpenAI | null

    # ----------------------------------------------------------
    # init() - 初始化：连接所有 MCP 服务器，收集工具，创建 LLM
    #
    # async def = 异步方法
    #   等价于 Java: public async CompletableFuture<Void> init() { ... }
    #   等价于 TS: async init(): Promise<void> { ... }
    #
    # Python 的 async/await 跟 JS 几乎一样：
    #   • async def 定义一个协程（coroutine）——可以暂停和恢复的函数
    #   • await 等待一个 async 操作完成
    #   • 每个 await 都是一个"可能在这里暂停，先去干别的"的点
    #
    # Java 对比：
    #   如果你用过 CompletableFuture + thenCompose，await 就是那个 thenCompose
    #   但写法更像同步代码（这就是所谓的"异步代码同步写法"）
    # ----------------------------------------------------------
    async def init(self) -> None:
        """
        初始化 Agent：连接 MCP 服务器 → 收集工具 → 创建 LLM 客户端。

        执行顺序：
          1. 打印标题
          2. 遍历所有 MCPClient，逐个初始化（连接各自的 MCP 服务器）
          3. 从所有客户端收集工具列表（flatMap 操作）
          4. 用收集到的工具创建 ChatOpenAI 实例
        """

        # ----- 打印标题 -----
        # 等价于 TS: logTitle('TOOLS')
        # 等价于 Java: Logger.info("===== TOOLS =====")
        logTile("TOOLS")

        # ----- 初始化所有 MCP 客户端 -----
        # Python 的 for 循环：
        #   for 变量 in 可迭代对象:
        #       循环体（缩进！）
        #
        # 跟 Java 的 for-each 对比：
        #   Java:  for (MCPClient client : this.mcpClients) { ... }
        #   Python: for client in self.mcp_clients: ...
        #
        # 注意：Python 用缩进而不是 {} 来表示代码块
        # 冒号 + 缩进 = Java 的 { }
        for client in self.mcp_clients:
            # await = 等待异步操作完成
            # client.init() 做的事情：启动子进程、建立 stdio 连接、握手、获取工具列表
            await client.init()

        # ----- flatMap + 类型转换：把所有客户端的工具合并到一个大列表 -----
        # TypeScript:
        #   const tools = this.mcpClients.flatMap(client => client.getTools())
        #
        # Python 实现方式：嵌套列表推导式
        #
        # ⚠️ 类型转换：MCP SDK 的 Tool 是强类型对象（有 .name .description .inputSchema），
        #    而 ChatOpenAI 的 Tool = dict（用 .get("name") 访问）。
        #    所以需要用 vars(tool) 或手动构建 dict 来转换。
        #    vars(obj) = 把对象的 __dict__ 属性返回（即把对象转成 dict）
        #
        # 拆解：
        #   for client in self.mcp_clients       ← 外层循环：遍历每个客户端
        #       for tool in client.get_tools()   ← 内层循环：遍历该客户端的每个工具
        #           vars(tool)                   ← 把 MCP Tool 对象转为 dict
        #
        # Java 对比（Stream API）：
        #   List<Map<String, Object>> tools = mcpClients.stream()
        #       .flatMap(client -> client.getTools().stream())
        #       .map(tool -> Map.of(
        #           "name", tool.getName(),
        #           "description", tool.getDescription(),
        #           "inputSchema", tool.getInputSchema()
        #       ))
        #       .collect(Collectors.toList());
        tools = [
            vars(tool)  # ← 关键！把 MCP Tool 对象 → dict
            for client in self.mcp_clients
            for tool in client.get_tools()
        ]
        self.register=ToolRegister()
        for client in self.mcp_clients:
            for t in client.get_tools():
                self.register.register(
                    ToolDef(name=t.name,description=t.description,input_schema=t.inputSchema),
                    MCPExecutor(client=client,name=t.name),
                )

        # ----- 创建 LLM 客户端 -----
        # 把 system_prompt、tools、context 注入 ChatOpenAI
        # 之后 LLM 就知道自己有哪些工具可用、应该扮演什么角色
        #
        # 这里没有 await —— ChatOpenAI 的构造函数是同步的
        # （它只是保存参数，真正的 API 调用在 chat() 时才发生）
        self.llm = ChatOpenAI(
            model=self.model,
            system_prompt=self.system_prompt,
            tools=tools,
            context=self.context,
        )

    # ----------------------------------------------------------
    # close() - 清理：关闭所有 MCP 客户端连接
    #
    # Java 对比：
    #   类似于实现 AutoCloseable 接口的 close() 方法，
    #   用在 try-with-resources 的 finally 块里。
    #
    # Python 中也可以实现 __aenter__ / __aexit__ 来支持
    # async with Agent(...) as agent: 语法，
    # 但这里保持简单，手动调用 close()。
    # ----------------------------------------------------------
    async def close(self) -> None:
        """
        关闭所有 MCP 客户端连接。

        Python 的 try/except 对比 Java 的 try/catch：
          Java:  try { ... } catch (Exception e) { e.printStackTrace(); }
          Python: try: ... except Exception: traceback.print_exc()

        这里没有 try/except，如果某个 client 的 cleanup 抛出异常，
        异常会向上传播。如果你想要"一个失败不影响其他"的行为：
          for client in self.mcp_clients:
              try:
                  await client.cleanup()
              except Exception:
                  pass  # 忽略错误，继续清理下一个
        """
        for client in self.mcp_clients:
            # cleanup() = 关闭 stdio 传输、ClientSession、子进程
            # 具体做了什么见 mcpCient.py 里 MCPClient.cleanup 的详细注释
            await client.cleanup()

    # ----------------------------------------------------------
    # invoke() - 核心方法：驱动 LLM ↔ 工具 的交互循环
    #
    # 这是 Agent 的心脏。下面用一张图解释整个流程：
    #
    #   ┌─────────────────────────────────────────────────────┐
    #   │                    invoke(prompt)                     │
    #   │                                                       │
    #   │  ① llm.chat(prompt)  ← 发送用户问题                 │
    #   │         │                                             │
    #   │         ▼                                             │
    #   │  ② LLM 返回 response                                 │
    #   │         │                                             │
    #   │         ├── response.toolCalls 有内容？               │
    #   │         │   YES → ③ 执行每个 toolCall                │
    #   │         │        ④ 把结果追加到对话上下文              │
    #   │         │        ⑤ llm.chat()  ← 让 LLM 继续         │
    #   │         │        ⑥ 回到 ②（循环）                    │
    #   │         │                                             │
    #   │         └── response.toolCalls 为空？                 │
    #   │             → ⑦ return response.content  ← 最终答案  │
    #   └─────────────────────────────────────────────────────┘
    #
    # 简单理解：
    #   LLM 就像一个人，你可以给他工具。
    #   你说"帮我查天气"，他会说"我需要调用 get_weather 工具"。
    #   你帮他执行 get_weather，把结果告诉他。
    #   他看了结果说"今天晴，25°C"。
    #   这就是 invoke() 做的事。
    # ----------------------------------------------------------
    #user_id暂时用"1"
    async def invoke(self, prompt: str,user_id:str="1", max_steps: int = 10) -> str:
        """
        执行一次交互：发送 prompt，循环处理工具调用，返回最终答案。

        参数：
          prompt : 用户输入的问题或指令

        返回值：
          LLM 的最终文字回复（str）

        异常：
          RuntimeError : 如果 Agent 还没初始化（self.llm 为 None）
        """

        # ----- 前置检查：LLM 初始化了没有？ -----
        # Python 的 if not xxx:
        #   等价于 Java 的 if (xxx == null)
        #   等价于 JS 的 if (!xxx)
        # None、空字符串、空列表、False、0 在布尔上下文中都是 False
        id=uuid.uuid4().hex[:12]
        token=trace_id.set(id)
        try:
            if self.llm is None:
                # raise = Java 的 throw
                # RuntimeError = 标准异常类型（类似 Java 的 IllegalStateException）
                raise RuntimeError(
                    "Agent 还没有初始化！请先调用 await agent.init()"
                )

            memory = MemoryManager() if user_id else None

            if memory:
                short = await memory.load_short_term(user_id)
                if short:
                    safe_print(f"[短期记忆] 加载摘要: {short[:80]}...")
                    self.llm.messages.append({
                        "role": "system",
                        "content": f"[用户上次对话摘要] {short}"
                    })

                long_docs = await memory.search_long_term(user_id,query=prompt,k=3)
                if long_docs:
                    safe_print(f"[长期记忆] 检索到 {len(long_docs)} 条相关历史")
                    self.llm.messages.append({
                        "role": "system",
                        "content": "[用户历史相关记录]\n" + "\n".join(
                            f"- {d[:200]}" for d in long_docs
                        )
                    })

            # ----- 第1步：发送 prompt 给 LLM -----
            # 等价于 TS: let response = await this.llm.chat(prompt);
            # LLM 可能直接返回文字答案，也可能返回 tool_calls（"我需要调用工具"）

            try:
                response = await self.llm.chat(prompt)
                self.token_budget.recount(self.token_budget.count_messages(self.llm.messages))
            except Exception as e:
                original = getattr(e, '__cause__', e) or e
                error_msg = str(original) or repr(original) or type(original).__name__
                return json.dumps(
                    {
                        "status":"error",
                        "component":"llm",
                        "message":error_msg,
                        "hint":"LLM服务多次重试后仍不可用，请检查API配置或稍后重试",
                    },
                    ensure_ascii=False
                )
            # ----- 第2步：Tool-use 循环 -----
            # while True = Java 的 while (true)
            # Python 没有 do...while，但 while True + break 可以模拟任何循环
            #给一个step，防止重复多次
            step=0
            while step<max_steps:
                if self.token_budget.exceeded:
                    return "[已达到 Token 预算上限，任务中断]"
                step+=1
                # tool_calls 是一个 list[dict]，每个元素形如：
                #   {
                #     "id": "call_abc123",
                #     "function": {
                #       "name": "read_file",
                #       "arguments": '{"path": "/tmp/hello.txt"}'
                #     }
                #   }
                #
                # Python 的 if xxx: 对列表来说：
                #   [] (空列表) → False → 跳过 if 块
                #   [item]      → True  → 进入 if 块
                if response["toolCalls"]:
                    # ----- 遍历每个 tool call -----
                    # Java: for (Map<String, Object> toolCall : response.getToolCalls())
                    for tool_call in response["toolCalls"]:
                        # 从 tool call 中取出函数名
                        # tool_call["function"]["name"] 等同于
                        # Java: toolCall.get("function").get("name")
                        func_name = tool_call["function"]["name"]
                        '''
                        # ----- 查找能处理这个工具调用的 MCP 客户端 -----
                        # Python 的 next() + 生成器表达式 = Java 的 Stream.findFirst()
                        #
                        # 拆解：
                        #   (client for client in self.mcp_clients ...)
                        #     ↑ 生成器表达式（generator expression）
                        #     类似于 Java Stream 的惰性求值——用的时候才计算
                        #
                        #   if any(t.name == func_name for t in client.get_tools())
                        #     ↑ any() = 有一个为 True 就返回 True
                        #     ↑ t.name == func_name for t in ... = 内部又是一个生成器
                        #
                        #   next(..., None) = 取第一个匹配的，没有就返回 None
                        #
                        # Java 对比：
                        #   Optional<MCPClient> mcp = mcpClients.stream()
                        #       .filter(c -> c.getTools().stream()
                        #           .anyMatch(t -> t.name.equals(funcName)))
                        #       .findFirst();
                        mcp = next(
                            (
                                client
                                for client in self.mcp_clients
                                if any(
                                    t.name == func_name
                                    for t in client.get_tools()
                                )
                            ),
                            None,  # 默认值：没找到就返回 None
                        )


                        if mcp is not None:
                            # ===== 有对应的 MCP 客户端 → 执行工具调用 =====
                            logTile("TOOL USE")

                            # ----- 日志输出 -----
                            # f-string（格式化字符串）：
                            #   f"hello {name}" = Java: "hello " + name
                            #   f"hello {name!r}" = 带 repr（调试输出）
                            safe_print(f"调用工具: {func_name}")
                            safe_print(f"参数: {tool_call['function']['arguments']}")

                            # ----- 解析参数 -----
                            # JSON.parse(toolCall.function.arguments) 的 Python 版
                            # json.loads() = JSON string → Python 对象
                            #
                            # Python ↔ JSON 对照：
                            #   JSON string  → Python str
                            #   JSON number  → Python int/float
                            #   JSON boolean → Python bool（注意大小写：True/False）
                            #   JSON null    → Python None
                            #   JSON array   → Python list
                            #   JSON object  → Python dict

                            params = json.loads(tool_call["function"]["arguments"])

                            # ----- 调用 MCP 工具 -----
                            # mcp.call_tool() 是异步的，所以用 await
                            # 返回值是 CallToolResult 对象（MCP SDK 的类型），不是 dict
                            # 需要用 .content 提取实际内容
                            try:
                                # 实验 1 已结束，注释掉模拟故障
                                # raise RuntimeError("模拟数据库连接超时")
                                result = await mcp.call_tool(func_name, params)
                                #result_text = '{"status": "error", "message": "数据库连接超时"}'

                                # ----- 提取 CallToolResult 中的文本内容 -----
                                # CallToolResult.content 是一个 list，每个元素有 .type 和 .text
                                # 对 LLM 来说只需要文字，所以把 text 类型的块拼接起来
                                result_text_parts: list[str] = []

                                for block in result.content:
                                    if hasattr(block, "text"):
                                        result_text_parts.append(block.text)
                                result_text = "\n".join(result_text_parts)
                            except Exception as e:
                                error_msg = str(e) or repr(e) or type(e).__name__
                                result_text=json.dumps(
                                    {
                                        "status":"error",
                                        "tool":func_name,
                                        "message":error_msg,
                                        "hint":"请尝试其他方式或告知用户当前无法完成此操作",
                                    },
                                    ensure_ascii=False,
                                )

                            # ----- 打印结果 -----
                            safe_print(f"结果: {result_text}")

                            # ----- 把工具结果追加到 LLM 对话上下文 -----
                            # 关键！LLM 需要看到工具执行结果才能继续推理
                            # 如果不调用 append_tool_result，LLM 就不知道工具返回了什么
                            self.llm.append_tool_result(
                                tool_call["id"],
                                result_text,
                            )

                        else:
                            # ===== 没有对应的 MCP 客户端 → 返回错误信息 =====
                            # 这也是一种"结果"——告诉 LLM 这个工具不存在
                            # LLM 看到后会尝试其他方式或报告给用户
                            self.llm.append_tool_result(
                                tool_call["id"],
                                "Tool not found",
                            )
                    '''
                        params = json.loads(tool_call["function"]["arguments"])
                        try:
                            tool_result=await self.register.execute(func_name,params)
                            result_text=tool_result.to_message()
                        except KeyError as e:
                            result_text = "Tool not found"
                        # ----- 打印结果 -----
                        safe_print(f"结果: {result_text}")

                        # ----- 把工具结果追加到 LLM 对话上下文 -----
                        # 关键！LLM 需要看到工具执行结果才能继续推理
                        # 如果不调用 append_tool_result，LLM 就不知道工具返回了什么
                        self.llm.append_tool_result(
                            tool_call["id"],
                            result_text,
                        )

                    # ----- 工具调用全部处理完毕，让 LLM 继续对话 -----
                    # 这次不带 prompt 参数——LLM 从对话历史中看到工具结果，
                    # 自动决定下一步：是继续调用工具？还是输出最终答案？
                    response = await self.llm.chat()
                    self.token_budget.recount(self.token_budget.count_messages(self.llm.messages))
                    # continue = 回到 while True 的开头，检查新一轮的 tool_calls
                    # Java: continue;
                    # 效果一样：跳过本轮循环剩余部分，进入下一次判断

                    if self.token_budget.should_compact():
                        await self.compact_old_messages(keep_recent=4)

                    continue

                # ----- 没有 tool_calls → LLM 给出了最终答案 -----
                # 跳出循环，准备返回结果
                break
            if memory:
                sys_msg = [m for m in self.llm.messages if m.get("role") == "system"]
                oth_msg = [m for m in self.llm.messages if m.get("role") != "system"]
                # 只总结整段对话，【不改】messages、不 mark_summarized
                summary = await self.summarize(sys_msg=sys_msg, to_be_summarize=oth_msg) # type: ignore
                if summary:
                    await memory.save_short_term(user_id, summary)
                    safe_print("[短期记忆] 摘要已保存")

                await memory.save_long_term(
                    question=prompt,
                    user_id=user_id,#暂时用“1”
                    answer=response["content"][:500],
                )

            # ----- 第3步：返回结果 -----
            return response["content"]
        finally:
            trace_id.reset(token)

    async def invoke_stream(self, prompt: str, max_steps: int = 10):
        id=uuid.uuid4().hex[:12]
        token=trace_id.set(id)
        try:

            """流式版 invoke:工具循环每一轮都走 chat_stream,边收边 yield。"""
            if self.llm is None:
                raise RuntimeError("Agent 还没有初始化！请先调用 await agent.init()")

            step = 0
            while step < max_steps:
                step += 1
                if self.token_budget.exceeded:
                    yield "[已达到 Token 预算上限，任务中断]"
                    return

                # 每轮重新攒,不残留上一轮的 tool_calls
                content = ""
                tool_calls: list[dict] = []

                # 第一轮带 prompt,后续轮不带(上下文已在 messages 里)
                async for delta in self.llm.chat_stream(prompt if step == 1 else ""):
                    if delta.content:
                        content += delta.content
                        yield delta.content
                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            while len(tool_calls) <= tc.index:
                                tool_calls.append({"id": "", "function": {"name": "", "arguments": ""}})
                            cur = tool_calls[tc.index]
                            if tc.id:
                                cur["id"] += tc.id
                            if tc.function and tc.function.name:
                                cur["function"]["name"] += tc.function.name
                            if tc.function and tc.function.arguments:
                                cur["function"]["arguments"] += tc.function.arguments

                # 收尾:append assistant(含 tool_calls)
                self.llm.append_assistant(content, tool_calls)

                if not tool_calls:
                    break  # 这轮没工具调用 → 结束

                # 执行这轮的工具
                for tool_call in tool_calls:
                    func_name = tool_call["function"]["name"]
                    params = json.loads(tool_call["function"]["arguments"])
                    try:
                        tool_result = await self.register.execute(func_name, params)
                        result_text = tool_result.to_message()
                    except KeyError:
                        result_text = "Tool not found"
                    safe_print(f"结果: {result_text}")
                    self.llm.append_tool_result(tool_call["id"], result_text)

                # 回 while 顶部,开新一轮(不带 prompt)
        finally:
            trace_id.reset(token)



    async def compact_old_messages(self,keep_recent:int=4):
        assert self.llm is not None
        messages=self.llm.messages
        sys_msg=[m for m in messages if m.get("role")=="system"]
        oth_msg=[m for m in messages if m.get("role")!="system"]

        if len(oth_msg)<=2+keep_recent:
            return ""

        middle=oth_msg[:-keep_recent]
        recent=oth_msg[-keep_recent:]

        if len(middle)<=2:
            return ""

        compacted=await self.summarize(sys_msg=sys_msg,to_be_summarize=middle) # type: ignore

        #新增摘要
        summary_msg={"role": "system", "content": f"[之前对话摘要] {compacted}"}
        self.llm.messages=sys_msg+[summary_msg]+recent # type: ignore
        #重新计算token
        self.token_budget.recount(self.token_budget.count_messages(self.llm.messages))
        self.token_budget.mark_summarized()

        return compacted

    async def summarize(self,sys_msg:list[dict],to_be_summarize:list[dict]):

        summary_prompt=(
            "请用 200 字以内总结以下对话的关键信息，包括：\n"
            "- 讨论了哪些话题\n"
            "- 用户问了什么问题、有什么需求\n"
            "- 重要的结论或知识点\n\n"
            "对话内容：\n"
            f"{json.dumps(to_be_summarize, ensure_ascii=False, default=str)}\n\n"
            "只返回摘要文本，不要加任何前缀。"
        )

        assert self.llm is not None
        saved_messages=self.llm.messages
        self.llm.messages=sys_msg+[{"role":"user","content":summary_prompt}] # type: ignore

        try:
            summary_response=await self.llm.chat()
            summary=summary_response["content"].strip()
        except Exception as e:
            safe_print(f"压缩失败:{e}")
            self.llm.messages=saved_messages
            return ""

        self.llm.messages=saved_messages

        return summary


# ============================================================
# 使用示例
#
# 下面是 Agent 的典型使用方式，注释掉了因为需要实际的 MCP 配置。
# 你可以取消注释并填入自己的配置来测试。
# ============================================================
# async def example() -> None:
#     """
#     示例：创建一个带文件系统工具的 Agent，让它读一个文件。
#
#     Python 中的 async def 对比 Java：
#       Java:  public static CompletableFuture<Void> example() { ... }
#       Python: async def example() -> None: ...
#
#       Python 的 -> None 表示"没有返回值"（类似 Java 的 void）。
#       不写 -> None 也不影响运行（因为 Python 是动态类型），
#       但写了可以让 IDE 更智能。
#     """
#
#     # 创建一个 MCP 客户端：连接文件系统服务器
#     filesystem_client = MCPClient(
#         name="filesystem",
#         command="npx",
#         args=["-y", "@modelcontextprotocol/server-filesystem", "."],
#     )
#
#     # 创建 Agent
#     agent = Agent(
#         model="deepseek-chat",
#         mcp_clients=[filesystem_client],
#         system_prompt="你是一个文件助手，帮用户读写文件。",
#     )
#
#     # 初始化（连接 MCP 服务器 + 收集工具）
#     await agent.init()
#
#     # 发送问题
#     answer = await agent.invoke("读取 README.md 的内容")
#     print(f"\n最终答案: {answer}")
#
#
# # Python 的程序入口
# # if __name__ == "__main__":  =  当直接运行此文件时执行
# # 如果这个文件被 import，下面的代码不会执行
# if __name__ == "__main__":
#     # asyncio.run() = 启动异步事件循环
#     # 相当于 Java 的: ExecutorService.submit(task).get()
#     import asyncio
#     asyncio.run(example())
async def example():
    filesystem_client = MCPClient(
        name="filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "."],
    )
    agent=Agent(
        model="deepseek-chat",
        mcp_clients=[filesystem_client],
        system_prompt="你是一个文件助手"
    )
    questions = [
        "读取并概况当前目录的source.*的文件",
    ]
    answer:list[str]=[]
    await agent.init()
    for q in questions:
        #answer.append(await agent.invoke(q,"1"))
        async for s in agent.invoke_stream(q):
            print(s,end="",flush=True)
    print()
    await agent.close()  # 显式清理，避免 npx 进程残留
    #print(f"\n最终答案: {answer}")

if __name__ == "__main__":
    #asyncio.run() = 启动异步事件循环
    #相当于 Java 的: ExecutorService.submit(task).get()
    import asyncio
    asyncio.run(example())