"""
MCPClient - MCP (Model Context Protocol) 客户端封装

这是参考 MCP 官方 quickstart 的 TypeScript 版本翻译的 Python 实现:
  https://modelcontextprotocol.io/quickstart/client

Python 新手阅读指南：
  1. 先看 class MCPClient 的 __init__（了解初始化做了什么）
  2. 再看 _connect_to_server（核心：如何连接 MCP 服务器）
  3. 最后看 call_tool 和 cleanup（如何使用和清理）
  4. 底部的 if __name__ == "__main__" 是程序入口
"""

import asyncio
import os
from typing import Any, Optional
from contextlib import AsyncExitStack
from anyio import ClosedResourceError


# ----- Windows 中文环境：强制 stdout 使用 UTF-8 -----
# 否则 rich 输出中文会 throw UnicodeEncodeError（GBK 无法编码）
# sys.stdout.reconfigure() 是 Python 3.7+ 的功能
# 万一不支持也不影响运行
'''if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass'''


# MCP SDK 的核心类
# ClientSession  : 与 MCP 服务器的会话，发送/接收消息都靠它
# StdioServerParameters : 描述如何启动一个 MCP 服务器进程（命令行 + 参数）
# Tool          : MCP 工具的数据类型（name, description, inputSchema）

from mcp import ClientSession, StdioServerParameters, Tool
from mcp.client.stdio import stdio_client  # stdio 传输层：通过标准输入输出通信

from rich import print as rprint  # rich 的 print：自带颜色和格式化

from dotenv import load_dotenv

from utils import logTile  # 项目内的日志小工具

load_dotenv()  # 加载 .env 文件中的环境变量

_DEVNULL = open(os.devnull, "w")   # os.devnull → "nul"，类型是 TextIO，IDE 不再报红


# ============================================================
# MCPClient - MCP 客户端
#
# 对应 TS 版本的 MCP Client 类。
# 核心职责：
#   1. 启动一个 MCP 服务器子进程
#   2. 通过 stdio（标准输入输出）与服务器通信
#   3. 列出服务器提供的工具
#   4. 调用工具并返回结果
# ============================================================
class MCPClient:
    """
    封装 MCP 协议的客户端。

    Python 核心概念速查：
    - self      : 相当于 JS/TS 的 this，但必须显式写在每个方法的第一个参数
    - __init__  : 构造函数，等价于 constructor()
    - async def : 异步方法，等价于 async function
    - await     : 等待异步操作，跟 JS 一样
    - type hint : 类型注解（如 list[Tool]），Python 3.10+ 原生支持泛型
    """

    # ----------------------------------------------------------
    # 构造函数
    #
    # Python 的 __init__ 就是构造函数（双下划线 = "dunder" = 魔法方法）
    # self 参数必须显式写出来，调用时不需要传（Python 会自动处理）
    # ----------------------------------------------------------
    def __init__(
        self,
        name: str,
        command: str,
        args: list[str],
        version: str = "0.0.1",
    ) -> None:
        """
        参数说明：
          name    : MCP 服务器名称（自定义，用于日志显示）
          command : 启动服务器的命令，如 "npx" 或 "python"
          args    : 命令行参数列表，如 ["-y", "@modelcontextprotocol/server-filesystem", "/path"]
          version : 协议版本号，一般不用改
        """

        # ----- 一、基础属性 -----
        self.name = name
        self.version = version
        self.command = command
        self.args = args

        # ----- 二、延迟初始化的属性 -----
        self.session: Optional[ClientSession] = None

        # ----- 三、AsyncExitStack：Python 的异步资源管理器 -----
        # 这是 Python 特有的概念，TS 中没有直接对应物。
        #
        # 理解 "资源管理"：
        #   当你打开一个文件、启动一个子进程、建立一个连接，
        #   你需要确保"用完一定关闭"，否则会泄漏资源。
        #
        # TS 的做法：手动 try/finally 或者使用 AbortController
        # Python 的做法：用 with（同步）/ async with（异步）语句
        #   with open("file.txt") as f:   # 离开 with 块自动 f.close()
        #       f.read()
        #
        # AsyncExitStack 是"动态版"的 async with ——
        # 当你不确定有多少个资源需要管理时，用它：
        #   stack.enter_async_context(资源)  # 注册资源
        #   await stack.aclose()             # 一次性关闭所有资源
        #
        # 简单理解：它是一个"资源清单"，退出时按注册的逆序逐个关闭。
        self.exit_stack = AsyncExitStack()

        # ----- 四、工具列表缓存 -----
        # 连接成功后从服务器获取，缓存在这里
        # list[Tool] 等价于 TS 的 Tool[]
        self.tools: list[Tool] = []

    # ----------------------------------------------------------
    # 初始化：连接到 MCP 服务器
    #
    # 把 init 和 _connect_to_server 分开是为了灵活性——
    # 你可以只创建实例而不连接，稍后再手动调用 init()
    # ----------------------------------------------------------
    async def init(self) -> None:
        """
        初始化客户端：连接服务器并获取工具列表。

        Python 的 async def：
        - 跟 JS 的 async function 几乎一样
        - 调用 async def 返回的是一个 coroutine（协程），需要用 await 等待
        - 顶层调用需要 asyncio.run() 来启动（见文件末尾）
        """
        await self._connect_to_server()

    # ----------------------------------------------------------
    # 清理资源
    #
    # Python 没有析构函数（destructor），因为 Python 用 GC（垃圾回收），
    # 对象的销毁时机是不确定的。所以需要显式调用 cleanup()。
    #
    # 另一种做法是实现 __aexit__ 让实例支持 async with：
    #   async with MCPClient(...) as client:
    #       await client.init()
    # 这里先用简单的手动 cleanup 方案。
    # ----------------------------------------------------------
    async def cleanup(self) -> None:
        """
        关闭所有资源：断开连接、终止子进程等。

        try / except Exception:
        - 跟 JS 的 try / catch 一样
        - Exception 是绝大多数异常的基类（类似 JS 的 Error）
        - 这里捕获所有异常是为了确保 cleanup 不会抛错中断程序
        """
        try:
            # aclose() 会按注册的逆序关闭所有在 exit_stack 中注册的资源
            # 包括：stdio 传输 -> ClientSession -> 子进程
            await self.exit_stack.aclose()
        except (Exception, asyncio.CancelledError):
            # ⚠️ Python 3.9+ 把 CancelledError 从 Exception 移到了 BaseException
            # 所以必须显式写在 except 列表里，否则抓不住
            # 即使清理出错也要继续，不能影响主流程
            rprint("Error during MCP client cleanup, traceback and continue!")
            import traceback
            traceback.print_exc()  # 打印堆栈但不中断

    # ----------------------------------------------------------
    # 获取工具列表
    # ----------------------------------------------------------
    def get_tools(self) -> list[Tool]:
        """
        返回服务器提供的工具列表。

        Python 的 @property 装饰器可以实现类似 TS 的 getter，
        但这里用简单的方法调用就够了，保持跟 TS 版本一致。
        """
        return self.tools

    # ----------------------------------------------------------
    # 核心方法：连接 MCP 服务器
    #
    # _ 前缀是 Python 约定："我是内部方法，外部别调"
    # Python 没有 private 关键字，全靠命名约定
    # ----------------------------------------------------------
    async def _connect_to_server(self) -> None:
        """
        启动 MCP 服务器子进程并建立 stdio 连接。

        流程概览：
          1. 配置启动参数（命令行 + 参数）
          2. 启动子进程，建立 stdio 传输通道
          3. 基于传输通道创建 ClientSession
          4. 发送 initialize 握手请求
          5. 调用 list_tools() 获取工具列表

        每一步都通过 exit_stack 注册，确保 cleanup 时能正确释放。
        """

        # ----- 第1步：配置服务器启动参数 -----
        # StdioServerParameters 描述了"怎么启动这个服务器进程"
        #
        # 等价于 TS:
        #   const serverParams = { command: "npx", args: ["-y", "..."] }
        #
        # TS 里这是匿名对象，Python 里是强类型的 dataclass 实例
        server_params = StdioServerParameters(
            command=self.command,
            args=self.args,
        )

        # ----- 第2步：建立 stdio 传输通道 -----
        # stdio_client(server_params) 做了三件事：
        #   1. 用 command + args 启动一个子进程
        #   2. 把子进程的 stdin/stdout 包装成 MCP 传输层
        #   3. 返回一个 async context manager（异步上下文管理器）
        #
        # async context manager 是什么？
        #   一个实现了 __aenter__ 和 __aexit__ 的对象，
        #   可以用 async with 来管理它的生命周期。
        #
        # enter_async_context() 做了什么？
        #   1. 调用 context_manager.__aenter__() → 启动子进程，拿到传输对象
        #   2. 把 context_manager.__aexit__() 注册到 exit_stack
        #      → 将来 aclose() 时会自动调用 __aexit__ 来关闭子进程
        #
        # TS 中没有这个机制，需要手动管理：
        #   const transport = await stdio_client(serverParams)
        #   // ... 用完需要手动关闭
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params, errlog=_DEVNULL),
        )

        # stdio_transport 是一个元组 (read_stream, write_stream)
        # Python 的 解包（unpacking） 等价于 TS 的 解构（destructuring）
        # TS: const [read, write] = transport
        # PY: self.stdio, self.write = stdio_transport
        self.stdio, self.write = stdio_transport

        # ----- 第3步：创建客户端会话 -----
        # ClientSession 需要 read_stream 和 write_stream 来收发 MCP 消息
        #
        # 同样通过 enter_async_context 注册，确保会话被正确关闭
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        # ----- 第4步：发送 initialize 握手请求 -----
        # MCP 协议规定：连接后必须先 initialize
        # 这类似于 HTTP 的 TLS 握手——双方确认协议版本和能力
        #
        # Python 类型窄化（type narrowing）：
        # self.session 类型是 Optional[ClientSession]（可以是 None），
        # 类型检查器会在这里报警告。虽然逻辑上它不可能为 None
        # （因为上面刚创建），但类型检查器不知道。
        # 我们用 assert 来告诉类型检查器"这里一定不是 None"。
        # assert = "我断言这是真的，如果不是就崩掉"
        # 等价于 TS 的非空断言: this.session!.initialize()
        assert self.session is not None
        await self.session.initialize()

        # ----- 第5步：获取工具列表 -----
        # 向服务器请求它提供的所有工具
        # 返回的 response.tools 是 list[Tool] 类型
        #
        # Tool 的属性：
        #   .name        : 工具名称（如 "read_file"）
        #   .description : 工具描述
        #   .inputSchema : 参数 schema（JSON Schema 格式）
        response = await self.session.list_tools()
        self.tools = response.tools

        # ----- 第6步：打印连接信息 -----
        # 列表推导式（list comprehension）:
        #   [表达式 for 变量 in 可迭代对象]
        # 等价于 JS:
        #   tools.map(tool => tool.name)
        rprint(
            "\nConnected to server with tools:",
            [tool.name for tool in self.tools],
        )

    async def _reconnect(self) -> None:
        """连接断开后重建：关旧资源 → 换新 exit_stack → 重新握手。"""
        await self.exit_stack.aclose()      # 杀掉旧的 node 子进程、关旧流
        self.exit_stack = AsyncExitStack()  # exit_stack 关闭后不能复用，必须换新的
        self.session = None
        self.tools = []
        await self._connect_to_server()     # 重新 spawn + initialize + list_tools


    # ----------------------------------------------------------
    # 调用工具
    #
    # 这是对外暴露的主要方法，让 LLM 可以通过 MCP 执行操作。
    # 比如 LLM 决定"读文件"，就会调用 call_tool("read_file", {"path": "/foo"})
    # ----------------------------------------------------------
    async def call_tool(self, name: str, params: dict[str, Any]) -> Any:
        """
        调用 MCP 服务器上的工具。

        参数：
          name   : 工具名称，如 "read_file"
          params : 工具参数，如 {"path": "/tmp/hello.txt"}

        返回值：
          工具执行结果，具体结构取决于工具定义。

        dict[str, Any] 类型注解：
          dict          = 字典（类似 JS 的 Object / Map）
          str           = key 必须是字符串
          Any           = value 可以是任意类型
          等价于 TS: Record<string, any>
        """
        if self.session is None:
            raise RuntimeError(
                "Session not initialized. Call init() first."
            )

        # session.call_tool() 发送 MCP 请求并等待结果
        # 底层走的是 JSON-RPC over stdio
        try:
            return await self.session.call_tool(name, params)
        except ClosedResourceError:
            rprint(f"[MCP] 连接断开，自动重连后重试: {name}")
            await self._reconnect()
            return await self.session.call_tool(name, params)



# ============================================================
# 示例：如何用 MCPClient 连接文件系统服务器
#
# 这个 example() 演示了 MCPClient 的完整使用流程：
#   1. 创建实例
#   2. 初始化（连接服务器）
#   3. 获取工具列表
#   4. 清理
#
# 运行方式：
#   python mcpCient.py
#   或者在别处 import 后：asyncio.run(example())
# ============================================================
async def example() -> None:
    """
    示例：连接 MCP 文件系统服务器，打印工具列表。

    Python 的类型提示 -> None：
      表示这个函数没有返回值（类似 TS 的 void）。
      Python 中不写 -> None 也没关系（因为是动态类型），
      但写了可以让 IDE 更智能地提示。
    """

    # 启动 filesystem MCP 服务器的配置
    # 这个例子用 npx 来运行 @modelcontextprotocol/server-filesystem
    # npx = Node.js 的包执行器（类似 npx）
    # -y   = 自动确认安装（类似 npx --yes）

    # 方式一：直接指定参数（推荐新手使用）
    # ⚠️ @modelcontextprotocol/server-filesystem 是 npm 包，必须用 npx 运行
    #   uvx = Python 包运行器（类似 pipx）
    #   npx = Node.js 包运行器
    mcp_client = MCPClient(
        name="filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "."],
    )

    # 方式二：从预设配置创建（等后面学到配置管理再讲）
    # from augmented.mcp_tools import PresetMcpTools
    # mcp_client = MCPClient(**PresetMcpTools.filesystem.to_common_params())

    # ----- 连接并获取工具 -----
    logTile("MCP CLIENT DEMO")

    await mcp_client.init()  # 连接服务器 + 获取工具列表

    # ----- 查看获取到的工具 -----
    tools = mcp_client.get_tools()
    rprint(f"\n获取到 {len(tools)} 个工具：")
    for tool in tools:
        # Python 的 f-string（格式化字符串）:
        #   f"前缀 + {变量}" = JS 的 `前缀 ${变量}`
        #   f"{tool.name:<20}" = 左对齐，宽度 20（类似 JS 的 padEnd）
        rprint(f"  - {tool.name:<25} | {tool.description}")

    # ----- 清理 -----
    await mcp_client.cleanup()


# ============================================================
# Python 程序的入口
#
# if __name__ == "__main__": 是 Python 的特殊写法：
#   - 当你直接运行这个文件：python mcpCient.py
#     → __name__ 被设为 "__main__"，条件成立，执行 example()
#   - 当这个文件被 import：
#     → __name__ 被设为模块名 "mcpCient"，条件不成立，不执行
#
# 这是 Python 区分"脚本模式"和"库模式"的标准写法。
# TS 中通常用不同的文件或 package.json scripts 来区分。
# ============================================================
if __name__ == "__main__":
    # asyncio.run() 是 Python 异步程序的启动器
    # 它创建一个事件循环（event loop），运行你的 async 函数，
    # 等它完成后自动清理。
    #
    # TS 中你可以在顶层直接 await（Node 18+ ESM），
    # Python 不行——await 必须在 async 函数内，
    # 顶层要用 asyncio.run() 来桥接。
    asyncio.run(example())
