import asyncio
import functools
from typing import Any, Awaitable, Callable, overload

from tool_result import ToolResult


AsyncToolFunc = Callable[..., Awaitable[Any]]


# 装饰器工厂有两种用法，用 @overload 分别声明返回类型：
#   @set_tool             → 传的是函数，直接返回包装后的 AsyncToolFunc
#   @set_tool(max_try=2)  → 不传函数，返回「装饰器」，再由 Python 拿去包函数
@overload
def set_tool(func: AsyncToolFunc) -> AsyncToolFunc: ...


@overload
def set_tool(
    func: None = None,
    *,
    max_try: int = 0,
    wait_time: float = 30.0,
) -> Callable[[AsyncToolFunc], AsyncToolFunc]: ...


def set_tool(
    func: AsyncToolFunc | None = None,
    *,
    max_try: int = 0,
    wait_time: float = 30.0,
) -> AsyncToolFunc | Callable[[AsyncToolFunc], AsyncToolFunc]:
    def decorator(fn: AsyncToolFunc) -> AsyncToolFunc:
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs) -> ToolResult:
            for attempt in range(0, max_try + 1):
                try:
                    raw = await asyncio.wait_for(fn(*args, **kwargs), wait_time)
                    return ToolResult(success=True, data=raw)
                except asyncio.TimeoutError:
                    if attempt == max_try:
                        return ToolResult(success=False, error="工具执行超时，请简化参数后重试")
                except Exception as e:
                    if attempt == max_try:
                        return ToolResult(success=False, error=f"工具执行失败:{type(e).__name__}")
                await asyncio.sleep(2 ** attempt)
            return ToolResult(success=False, error="未知错误")
        return wrapper

    if func is not None:
        return decorator(func)
    return decorator
