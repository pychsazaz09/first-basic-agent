"""
ChatOpenAI - 封装 OpenAI 兼容 API 的聊天客户端

这是从 TypeScript 版本翻译过来的 Python 实现。
Python 新手阅读指南：
  1. 从 class ChatOpenAI 开始看（核心逻辑）
  2. import 部分可以先跳过去，用到什么查什么
  3. 重点关注 self、async/await、类型注解 三个概念
"""

import os
from typing import Optional, cast

# Python 的包管理：从第三方库 import
# 等价于 TS 的 import OpenAI from "openai"
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from openai.types.shared_params.function_definition import FunctionDefinition

# 导入项目内的工具函数
# 等价于 TS 的 import { logTitle } from "./utils"
from utils import logTile


# ============================================================
# 类型别名
# Python 中用 type alias 代替 TS 的 interface
# Tool = Dict[str, Any]  等价于  type Tool = { [key: string]: any }
# 但为了跟 MCP SDK 的 Tool 类型对齐，暂时先用 dict
# ============================================================
Tool = dict


class ChatOpenAI:
    """
    封装 OpenAI 兼容 API 的聊天客户端。

    核心概念：
    - self     : 相当于 JS/TS 的 this，但 Python 要求显式声明
    - __init__ : 构造函数，相当于 constructor()
    - async def: 异步方法，等价于 async function
    - await    : 等待异步操作完成，跟 JS 一样
    """

    # ----------------------------------------------------------
    # 构造函数
    # Python 用 __init__ 而不是 constructor
    # 第一个参数 self 是必须的——它就是当前实例（类似 TS 的 this）
    # ----------------------------------------------------------
    def __init__(
        self,
        model: str,
        system_prompt: str = "",
        tools: Optional[list[Tool]] = None,
        context: str = "",
    ) -> None:
        """
        参数说明：
          model         : 模型名称，如 "deepseek-chat"
          system_prompt : 系统提示词（可选）
          tools         : MCP 工具列表（可选），None 表示没有工具
          context       : 初始用户上下文（可选）
        """

        # ----- 一、初始化 OpenAI 客户端 -----
        # 等价于 TS: new OpenAI({ apiKey, baseURL })
        # os.getenv("KEY") 等价于 TS: process.env["KEY"]
        self.llm = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )

        # ----- 二、保存模型名 -----
        # Python 没有 private 关键字，约定用 _ 前缀表示"别碰"
        self.model = model

        # ----- 三、处理 tools 默认值 -----
        # ⚠️ Python 大坑：默认参数只计算一次！
        # 所以不能用 tools: list[Tool] = [] （所有实例会共享同一个 list）
 #111       # 正确做法：tools=None，然后在函数体内判断
        self.tools: list[Tool] = tools if tools is not None else []

        # ----- 四、初始化消息列表 -----
        # list[X] 是泛型语法，等价于 TS 的 X[]
 #111       # ChatCompletionMessageParam 是 SDK 定义的消息类型
        self.messages: list[ChatCompletionMessageParam] = []

        # ----- 五、添加初始消息 -----
        # Python 用字典 {"key": "value"} 代替 JS 对象 { key: "value" }
        # 必须用双引号！（单引号也可以，但 JSON 风格推荐双引号）
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})
        if context:
            self.messages.append({"role": "user", "content": context})

    # ----------------------------------------------------------
    # 私有方法：把 MCP Tool 转换成 OpenAI 格式
    #
    # Python 没有 private/protected 关键字
    # _ 前缀是约定："我是内部实现，外部别调"
    # ----------------------------------------------------------
    def _get_tools_definition(self) -> list[ChatCompletionToolParam]:
        """
        遍历 self.tools，生成 OpenAI API 需要的 tools 参数格式。

        TS 版本:
          return this.tools.map(tool => ({
            type: "function",
            function: {
              name: tool.name,
              description: tool.description,
              parameters: tool.inputSchema,
            },
          }));
#11
        Python 版本用的是 列表推导式（list comprehension）:
          [表达式 for 变量 in 可迭代对象]
        这是 Python 最常用的写法之一，相当于 JS 的 .map()
        """
        return [
            ChatCompletionToolParam(
                type="function",
                function=FunctionDefinition(
                    name=t.get("name", ""),
                    description=t.get("description", ""),
                    parameters=t.get("inputSchema", {}),
                ),
            )
            for t in self.tools  # ← 列表推导式，等价于 .map()
        ]

    # ----------------------------------------------------------
    # 核心方法：发送消息并流式接收回复
    #
    # async def = JS 的 async function
    # 返回类型 -> dict 表示返回一个字典
    # ----------------------------------------------------------
    async def chat(self, prompt: str = "") -> dict:
        """发送 prompt，流式返回 { content, toolCalls }"""

        # ----- 打印标题 -----
        logTile("CHAT")  # 等价于 TS: logTitle('CHAT')

        # ----- 添加用户消息 -----
        if prompt:
            self.messages.append({"role": "user", "content": prompt})

        # ----- 准备 tools 参数 -----
        # 空列表不传 tools（避免 API 报错）
        tools_def = self._get_tools_definition()

        # ----- 发起流式请求 -----
        # Python 的类型检查器比较严格，这里用 if/else 分两路处理
        # 有 tools 和没有 tools 走不同的 create() 重载
        if tools_def:
            stream = await self.llm.chat.completions.create(
                model=self.model,
                messages=self.messages,
                stream=True,
                tools=tools_def,
            )
        else:
            stream = await self.llm.chat.completions.create(
                model=self.model,
                messages=self.messages,
                stream=True,
            )

        # ----- 初始化收集变量 -----
        content = ""
        tool_calls: list[dict] = []

        # ----- 流式读取 response -----
        logTile("RESPONSE")

        # async for = JS 的 for await...of
        # 逐块读取流式输出
        async for chunk in stream:
            delta = chunk.choices[0].delta  # 取第一个 choice 的 delta

            # --- 处理普通文本内容 ---
            if delta.content:
                content += delta.content
                # end='' 表示不换行，flush=True 表示立即输出（流式效果）
                print(delta.content, end="", flush=True)

            # --- 处理函数调用（Tool Call）---
            # delta.tool_calls 可能为 None 或空列表
            if delta.tool_calls:
                for tool_chunk in delta.tool_calls:
                    # 如果 tool_calls 还没存过这个 index，先创建一个坑位
                    # 等价于 TS: if (toolCalls.length <= toolCallChunk.index)
                    while len(tool_calls) <= tool_chunk.index:
                        tool_calls.append(
                            {
                                "id": "",
                                "function": {"name": "", "arguments": ""},
                            }
                        )

                    # 取出当前这个 tool call 的累积结果
                    current = tool_calls[tool_chunk.index]

                    # 流式累积：每次 chunk 可能只包含 id/name/arguments
                    # 的一部分，所以要 += 拼接
                    if tool_chunk.id:
                        current["id"] += tool_chunk.id
                    if tool_chunk.function and tool_chunk.function.name:
                        current["function"]["name"] += tool_chunk.function.name
                    if tool_chunk.function and tool_chunk.function.arguments:
                        current["function"]["arguments"] += (
                            tool_chunk.function.arguments
                        )

        # ----- 换行（流式输出结束）-----
        print()

        # ----- 把 assistant 的回复加入消息历史 -----
        # cast() 告诉类型检查器："我知道这是什么类型，相信我"
        # 因为 dict 是动态类型，需要手动收窄（narrowing）
        assistant_msg: dict = {"role": "assistant", "content": content}
        # 如果有 tool calls，也要加上
        if tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": c["id"],
                    "type": "function",
                    "function": c["function"],
                }
                for c in tool_calls
            ]
        self.messages.append(cast(ChatCompletionMessageParam, assistant_msg))

        # ----- 返回结果 -----
        return {
            "content": content,
            "toolCalls": tool_calls,
        }

    # ----------------------------------------------------------
    # 把工具执行结果追加到消息历史
    #
    # role: "tool" 告诉模型"这是你刚才调用的工具返回的结果"
    # ----------------------------------------------------------
    def append_tool_result(self, tool_call_id: str, tool_output: str) -> None:
        """追加工具调用结果到对话上下文"""
        self.messages.append(
            cast(ChatCompletionMessageParam, {
                "role": "tool",
                "content": tool_output,
                "tool_call_id": tool_call_id,
            })
        )
