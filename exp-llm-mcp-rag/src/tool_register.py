from typing import Any, Awaitable, Callable
from dataclasses import dataclass

from tool_result import ToolResult
from executor import LocalExecutor,MCPExecutor

AsyncToolFunc=Callable[...,Awaitable[Any]]

@dataclass
class ToolDef:
    name:str
    description:str
    input_schema:dict
    #func:AsyncToolFunc


class ToolRegister:
    def __init__(self) -> None:
        self._tools:dict[str,tuple[ToolDef,LocalExecutor|MCPExecutor]]={}

    def register(self,tool:ToolDef,executor:LocalExecutor|MCPExecutor)->None:
        self._tools[tool.name]=(tool,executor)

    def get_all_schemas(self)->list[dict]:
        return [
            {
                "type":"function",
                "function":{
                    "name":tool.name,
                    "description":tool.description,
                    "parameters":tool.input_schema,
                }
            }
            for tool,_ in self._tools.values()
        ]

    def list_names(self):
        return [name for name in self._tools.keys()]

    async def execute(self,name:str,args:dict)->ToolResult:
        _,executor=self._tools[name]
        return await executor.run(args)
