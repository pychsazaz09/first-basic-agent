import sys
from contextvars import ContextVar

# ----- 修复 Windows 终端编码问题 -----
# 在 Python 进程启动时提前把 stdout 设为 UTF-8。
# 这必须在任何 print() 之前执行，否则 GBK 终端会抛 OSError。
# mcpCient.py 中也有同样的逻辑（它先被导入，所以会先执行），
# 但 utils.py 内部调用 print()，保留自己的兜底。
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

trace_id=ContextVar("trace_id",default="")

def safe_print(*args, **kwargs):
    """
    print() 的安全包装：在 Windows 非 UTF-8 终端上自动回退。

    为什么需要这个函数？
      Windows 传统控制台（PowerShell/cmd 默认）使用 GBK 代码页。
      当 sys.stdout 被 mcpCient.py 切换为 UTF-8 后，WriteFile API
      向非 UTF-8 控制台写入可能触发 OSError [Errno 22]。
      此函数捕获该错误并回退到 ASCII 安全输出。
    """
    try:
        print(f"{trace_id.get()}",*args, **kwargs)
    except OSError:
        # 回退：把所有参数转为 repr（ASCII 安全），用 print 输出
        fallback_args = []
        for a in args:
            try:
                fallback_args.append(str(a))
            except Exception:
                fallback_args.append(repr(a))
        print(*fallback_args, **kwargs)


def logTile(message: str):
    """打印带分隔线的标题（如 ======== CHAT ========）"""
    total_length = 80
    msg_len = len(message)
    padding = max(0, total_length - msg_len - 4)
    left_pad = padding // 2
    right_pad = (padding + 1) // 2
    padded_msg = f"{'=' * left_pad} {message} {'=' * right_pad}"
    safe_print(padded_msg)