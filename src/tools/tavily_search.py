import httpx
from langchain.tools import tool
from src.config import settings

TAVILY_API_URL = "https://api.tavily.com/search"


@tool(description="Search the web using Tavily (optimized for AI agents, provides accurate and factual results)")
def search_web(query: str) -> str:
    try:
        response = httpx.post(
            TAVILY_API_URL,
            json={"query": query, "max_results": 5, "search_depth": "basic"},
            headers={"Authorization": f"Bearer {settings.tavily_api_key}"},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])
        if not results:
            return "未找到相关搜索结果"
        lines = ["🔍 Tavily 搜索结果：\n"]
        for i, item in enumerate(results, 1):
            title = item.get("title", "无标题")
            url = item.get("url", "")
            content = item.get("content", "")
            lines.append(f"{i}. {title}")
            if content:
                lines.append(f"   {content}")
            if url:
                lines.append(f"   链接：{url}")
            lines.append("")
        return "\n".join(lines)[:3000]
    except Exception as e:
        return f"Tavily 搜索失败：{e}"
