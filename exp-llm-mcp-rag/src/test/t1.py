import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_agent import RAGAgent


async def main() -> None:
    agent = RAGAgent(
        model="glm-5.3-flash",
        mcp_clients=[],
        system_prompt="你是一个严谨的文档助手。",
    )

    await agent.init()
    try:
        report = await agent.index_documents(r"D:\documents")
        print("索引结果：", report)
        for failure in report.failures:
            print("索引失败：", failure.source, failure.error)

        # 先直接检查检索，不让 LLM 掩盖召回问题。
        hits = await agent.search_knowledge("合同付款条件是什么？")
        for hit in hits:
            print(hit.citation, hit.rrf_score, hit.rerank_score)

        # 最后才测试 Agent 是否会调用 search_knowledge_base。
        answer = await agent.invoke("根据知识库说明合同付款条件，并引用来源")
        print(answer)

        unrelated_query = "知识库是否记载了超导量子比特的退相干时间？"
        unrelated_hits = await agent.search_knowledge(unrelated_query)
        print("无关问题命中数：", len(unrelated_hits))
        unrelated_answer = await agent.invoke(
            f"仅根据知识库回答：{unrelated_query}"
        )
        print(unrelated_answer)
    finally:
        await agent.close()

asyncio.run(main())