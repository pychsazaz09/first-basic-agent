# exp-llm-mcp-rag
Exp(loration)/(Exp)eriment project from https://github.com/KelvinQiu802/llm-mcp-rag , but impl in python

# 链路追溯
一、整体对齐：OpenAI 类型定义 + MCP 协议 + Agent 工具调用完整参数映射
先明确你用到的三类 TS/Pydantic 类型（OpenAI 兼容规范）：
FunctionDefinition：工具函数元信息（告诉 LLM「工具能干什么、参数结构」）
ChatCompletionToolParam：外层工具包装，type="function"，承载 FunctionDefinition；就是请求入参 tools[] 的单条结构
ChatCompletionMessageParam：对话消息单元（system/user/assistant/tool），包含工具调用入、出消息
同时融合 MCP（Model Context Protocol） 逻辑：
MCP Client 提供 get_tools() 获取服务端工具列表 → 把 MCP 工具定义转换成标准 OpenAI ChatCompletionToolParam
LLM 产出 tool_calls → 根据工具名称路由到对应 MCP Client → 调用 MCP 工具 → 将结果组装成 role="tool" 的消息丢回 LLM
1. 类型结构拆解（参数层级对应）
① FunctionDefinition
纯函数描述，核心是名称、说明、参数 JSON Schema
python
运行
# Pydantic 模拟定义（等价 openai 类型）
class FunctionDefinition(BaseModel):
    name: str
    description: str | None
    parameters: dict  # JSON Schema
② ChatCompletionToolParam
包裹 FunctionDefinition，对应请求 tools: List[ChatCompletionToolParam]
python
运行
class ChatCompletionToolParam(BaseModel):
    type: str = "function"
    function: FunctionDefinition
对应关系：MCP Tool → FunctionDefinition → ChatCompletionToolParam
plaintext
MCP Tool(name, description, inputSchema)
        ↓ 映射
FunctionDefinition(name, description, parameters=inputSchema)
        ↓ 包裹
ChatCompletionToolParam(type="function", function=FunctionDefinition)
③ ChatCompletionMessageParam
消息四大家族，完整闭环：
python
运行
# 1. user/system 普通消息
{"role": "user", "content": "xxx"}
# 2. assistant 消息（两种形态：纯文本 / 工具调用）
{
  "role": "assistant",
  "content": None,
  "tool_calls": [
      {
          "id": "call-xxx",
          "type": "function",
          "function": {"name": "tool_name", "arguments": "{...}"}
      }
  ]
}
# 3. tool 返回消息（工具执行结果，必须携带 tool_call_id）
{
  "role": "tool",
  "tool_call_id": "call-xxx",
  "content": str
}
2. 完整数据流（MCP Client + LLM 流式 ToolCall 参数链路）
plaintext
1. 【启动阶段】遍历 self.mcp_clients
   client.get_tools() → List[MCP Tool]
   MCP Tool → 转换 → ChatCompletionToolParam
   汇总所有工具，组成请求入参 tools: List[ChatCompletionToolParam]

2. 【LLM 请求】
messages: List[ChatCompletionMessageParam]
stream=True, tools=tools

3. 【流式接收 chunk.delta】
delta.tool_calls 增量分片 → 使用 index 组装完整 tool_calls

4. 【路由工具（你之前那段next()代码登场）】
根据 tool_call.function.name
next(client for client in mcp_clients if any(t.name == func_name for t in client.get_tools()))
找到提供该工具的 MCP Client

5. 【调用MCP工具】
入参：json.loads(arguments) → dict
result = await client.call_tool(name=tool_name, arguments=args_dict)

6. 【封装返回消息】tool类型 ChatCompletionMessageParam
{
    "role": "tool",
    "tool_call_id": tool_call["id"],
    "content": json.dumps(result.content, ensure_ascii=False)
}

7. 将这条tool消息追加进 messages，继续循环LLM请求（Agent多轮循环）


