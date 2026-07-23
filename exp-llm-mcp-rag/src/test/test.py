"""
Agent 集成测试（文件系统版）

用 filesystem MCP 服务器测试 Agent 的完整流程：
  Agent → ChatOpenAI → MCPClient → 文件系统工具

运行前：
  1. 确保 .env 在项目根目录（OPENAI_API_KEY + OPENAI_BASE_URL）
  2. 确保已安装 npx（Node.js 自带）

运行：
  cd exp-llm-mcp-rag && python src/test/test.py
"""

import sys
import asyncio
import os
from pathlib import Path

# ---- 路径设置 ----
current_file = Path(__file__)
src_dir = current_file.parent.parent  # test/ → src/
sys.path.insert(0, str(src_dir))

# ---- 加载环境变量 ----
project_root = current_file.parent.parent.parent
from click import prompt
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from mcpCient import MCPClient
from agent import Agent
from utils import logTile
from EmbeddingRetriever import EmbeddingRetriever

#https://jsonplaceholder.typicode.com/users
async def run():
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ 请先在 .env 里配置 OPENAI_API_KEY")
        return

    # MCP filesystem 的安全机制：
    #   最后一个参数是"允许访问的目录"，可以传多个
    #   不在列表里的路径 → 拒绝访问（沙箱隔离）
    #   这里把项目根目录的绝对路径传进去，确保一定被允许
    allowed_dir = str(project_root.resolve())  #111111111111

    # fetch: 网页抓取工具（mcp-server-fetch 已通过 uv tool install 预装）
    fetch = MCPClient("fetch", "mcp-server-fetch", [])

    # file: 文件系统工具
    file = MCPClient(
        name="filesystem",
        command="npx",
        args=[
            "-y",
            "@modelcontextprotocol/server-filesystem",
            allowed_dir,
        ],
    )
    current=os.path.dirname(os.path.abspath(__file__))
    #prompt:str=f"爬取这个网址https://jsonplaceholder.typicode.com/users里每个人信息，把每个人信息写入{now}，一人一份md"
    prompt:str=f"根据Bret的消息，写一个包含他信息的小故事，写在{current}这个目录下"
    context=await retrieve_context(prompt)
    agent = Agent(
        model="deepseek-v4-pro",
        mcp_clients=[fetch, file],   # ← 两个工具都给他
        system_prompt=f"你是一个文件助手。所有文件操作必须在 {allowed_dir} 目录下进行。",
        context=context
    )
    await agent.init()

    # 测试：
    response = await agent.invoke(prompt=prompt)
    print(f"\n===== 最终回答 =====\n{response}")

    await agent.close()

async def retrieve_context(task: str) -> str:
    """RAG：从 knowledge 目录加载文档，检索与 task 最相关的上下文。"""
    embedding_retriever = EmbeddingRetriever("BAAI/bge-m3")

    # 遍历 knowledge 目录，逐文件嵌入并存入向量库
    knowledge_dir = str(Path(__file__).parent / "knowledge")
    for file in os.listdir(knowledge_dir):
        file_path = os.path.join(knowledge_dir, file)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        await embedding_retriever.embed_document(content)

    # 检索与 task 最相关的 top 3 文档，拼接成一个字符串
    context = "\n".join(await embedding_retriever.retrieve(task, 3))

    logTile("CONTEXT")
    print(context)
    return context


if __name__ == "__main__":
    asyncio.run(run())
