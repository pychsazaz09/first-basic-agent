from tool_result import ToolResult

class LocalExecutor:
    def __init__(self,func) -> None:
        self.func=func

    async def run(self,args:dict)->ToolResult:
        return await self.func(**args)

class MCPExecutor:
    def __init__(self,client,name:str) -> None:
        self.client=client
        self.name=name

    async def run(self,args:dict)->ToolResult:
        try:
            raw=await self.client.call_tool(self.name,args)
            data=""
            result_text_parts: list[str] = []
            for block in raw.content:
                if hasattr(block,"text"):
                    result_text_parts.append(block.text)
            data="\n".join(result_text_parts)
            return ToolResult(success=True,data=data)
        except Exception as e:
            return ToolResult(success=False,error=f"MCP工具调用失败:{e}")
