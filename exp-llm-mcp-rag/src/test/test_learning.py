from modern_rag.models import IndexFailure,IndexReport
from modern_rag.models import IndexFailure, IndexReport

import pytest
from pathlib import Path
import asyncio
from rag_agent import RAGAgent


def test_report_is_ok_without_failures():
    report = IndexReport()
    assert report.ok is True



def test_report_is_not_ok_without_failures():

    failure=IndexFailure(source="test_source", error="解析失败")
    report=IndexReport(failures=[failure])
    assert report.ok is False

def test_temp_path_create_isolated_file(tmp_path:Path):
    path=tmp_path / "message.txt"
    path.write_text("hello",encoding="utf-8")
    content=path.read_text(encoding="utf-8")
    assert content=="hello"

def test_async_function_error_0():
    agent=RAGAgent(
            model="test",
            mcp_clients=[],
        )
    with pytest.raises(ValueError,match="top_k"):
        asyncio.run(agent.search_knowledge("hhh",0))

def test_async_function_error_21():
    agent=RAGAgent(
            model="test",
            mcp_clients=[],
        )
    with pytest.raises(ValueError,match="top_k"):
        asyncio.run(agent.search_knowledge("hhh",21))

def test_replace_real_api(monkeypatch):
    agent=RAGAgent(
            model="test",
            mcp_clients=[],
        )
    async def fake_search_knowledge(query:str,top_k:int):
        assert query=="不存在的问题"
        assert top_k==3
        return []
    monkeypatch.setattr(
        agent,
        "search_knowledge",
        fake_search_knowledge
    )
    result=asyncio.run(agent._search_knowledge_base("不存在的问题",3))
    assert result.success is True
    assert result.data is not None

    data = result.data
    assert data["evidence_count"] == 0
    assert data["insufficient_evidence"] is True
    assert data["matches"] == []