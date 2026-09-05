import asyncio
import json

from set_tool import set_tool
from tool_register import ToolDef, ToolRegister
from executor import LocalExecutor,MCPExecutor

from types import SimpleNamespace

class FakeMcpClient:
    async def call_tool(self, name: str, params: dict):
        return SimpleNamespace(
            content=[SimpleNamespace(text=f"[MCP:{name}] 收到参数 {params}")]
        )





# ========== 1. 定义真实工具：干活的异步函数，返回「原始数据」（不是 ToolResult） ==========

@set_tool(max_try=2, wait_time=5.0)   # 带参数：最多重试 2 次，单次超时 5 秒
async def get_weather(city: str) -> str:
    """查天气（假数据，代替真实的 HTTP 请求）。"""
    db = {"北京": "晴 25℃", "上海": "多云 28℃", "深圳": "雷阵雨 30℃"}
    return db.get(city, f"没查到 {city} 的天气")


@set_tool                            # 无括号：max_try=0, wait_time=30 默认
async def add(a: float, b: float) -> float:
    """两个数相加，安全无害。"""
    return a + b


@set_tool(max_try=0, wait_time=0.1)  # 故意让它超时，演示失败路径
async def slow_tool() -> str:
    await asyncio.sleep(1)
    return "太慢了，来不及了"


# ========== 2. 给每个工具配 JSON Schema（喂给 LLM 的「说明书」） ==========

def build_registry() -> ToolRegister:
    registry = ToolRegister()

    registry.register(ToolDef(
        name="get_weather",
        description="查询某个城市的天气，返回气温和天气状况",
        input_schema={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名，如 北京"},
            },
            "required": ["city"],
        }
    ),LocalExecutor(get_weather),)

    registry.register(ToolDef(
        name="add",
        description="两个数字相加",
        input_schema={
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "加数 a"},
                "b": {"type": "number", "description": "加数 b"},
            },
            "required": ["a", "b"],
            }
        ),
        LocalExecutor(add),
    )

    registry.register(ToolDef(
        name="slow_tool",
        description="一个很慢的工具，用来演示超时失败",
        input_schema={"type": "object", "properties": {}},
    ),LocalExecutor(slow_tool))

    mcp = FakeMcpClient()
    registry.register(
        ToolDef(name="read_file", description="读远端文件", input_schema={}),
        MCPExecutor(mcp, "read_file"),       # ← MCP 执行器，不是 LocalExecutor
    )

    return registry


# ========== 3. 演示：LLM 视角 vs 执行视角 ==========

async def main():
    registry = build_registry()

    # 3.1 LLM 视角：拿到所有工具的 schema（拼进 API 的 tools 参数）
    print("== 喂给 LLM 的 schemas ==")
    print(json.dumps(registry.get_all_schemas(), ensure_ascii=False, indent=2))
    print("\n已注册工具：", registry.list_names())

    # 3.2 执行视角：按名取工具调 func（LLM 说「调 get_weather(city=北京)」之后你做的事）
    print("\n== 执行结果 ==")
    r1 = await registry.execute("get_weather", {"city": "北京"})
    print("get_weather ->", r1.to_message())

    r2 = await registry.execute("add", {"a": 1, "b": 2.5})
    print("add         ->", r2.to_message())

    r3 = await registry.execute("slow_tool", {})
    print("slow_tool   ->", r3.to_message())  # 应该是 success=False

    r4 = await registry.execute("read_file", {"path": "/etc/hosts"})
    print("read_file   ->", r4.to_message())  # 应该是 success=False


if __name__ == "__main__":
    asyncio.run(main())
