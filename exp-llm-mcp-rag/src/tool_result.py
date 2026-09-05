from dataclasses import dataclass,field
from typing import Any
import json

@dataclass
class ToolResult:
    success:bool
    data:Any|None=None
    error:str|None=None

    def to_message(self):
        if self.success:
            return json.dumps({"status":"ok","data":self.data},ensure_ascii=False)
        return json.dumps({"status":"error","error":self.error},ensure_ascii=False)