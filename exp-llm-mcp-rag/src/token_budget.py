import tiktoken
from collections.abc import Sequence
from typing import Any

class TokenBudget:
    def __init__(self,model:str,budget:int) -> None:
        self.encoding=tiktoken.get_encoding(model)
        self.budget=budget
        self.used=0
        self.summarized=False

    def count(self,text:str):
        return len(self.encoding.encode(text))

    def count_messages(self,messages:Sequence[Any]):
        total=0
        for message in messages:
            content=message.get("content","")
            if isinstance(content,str):
                total+=self.count(content)
            if message.get("tool_calls"):
                for tc in message["tool_calls"]:
                    total+=self.count(tc.get("function",{}).get("arguments",""))
        return total

    def recount(self,token:int):
        self.used=token

    @property
    def near_limit(self):
        return self.used>=self.budget*0.8

    @property
    def exceeded(self):
        return self.used>=self.budget

    def should_compact(self):
        return self.near_limit and not self.summarized

    def mark_summarized(self):
        self.summarized=True
