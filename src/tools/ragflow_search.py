import logging

import httpx
from langchain.tools import tool
from src.config import settings
from src.rag.query_expander import expand_query

logger = logging.getLogger(__name__)


@tool(description="Search the RAGFlow knowledge base for professional travel information, guides, and destination details")
def search_knowledge_base(query: str) -> str:
    expanded = expand_query(query)
    if len(expanded) <= 1:
        result = _search_single(expanded[0])
        if result and "未找到" not in result:
            return f"[权威知识库来源，以下内容来自本地专业知识库，必须优先采用]\n{result}"
        return result

    logger.info("RAGFlow 多查询扩展检索：原始查询='%s'，扩展为 %d 个子查询", query, len(expanded))
    all_parts = []
    seen = set()
    found_any = False

    for sub_query in expanded:
        result = _search_single(sub_query)
        if result is None or "未找到" in result or "失败" in result or "出错" in result:
            continue
        deduped = []
        for line in result.split("\n"):
            if line not in seen:
                seen.add(line)
                deduped.append(line)
        deduped_str = "\n".join(deduped).strip()
        if deduped_str:
            all_parts.append(f"=== {sub_query} ===\n{deduped_str}\n")
            found_any = True

    if found_any:
        combined = "\n".join(all_parts).strip()[:3000]
        return f"[权威知识库来源，以下内容来自本地专业知识库，必须优先采用]\n{combined}"

    logger.warning("所有扩展查询均未返回结果，兜底使用原始查询")
    result = _search_single(query)
    if result and "未找到" not in result:
        return f"[权威知识库来源，以下内容来自本地专业知识库，必须优先采用]\n{result}"
    return result or "RAGFlow 知识库中未找到相关信息。"


def _search_single(query: str) -> str:
    try:
        url = f"{settings.ragflow_base_url}/api/v1/retrieval"
        response = httpx.post(
            url,
            json={"question": query, "dataset_ids": [settings.ragflow_kb_id], "page": 1, "size": 10},
            headers={"Authorization": f"Bearer {settings.ragflow_api_key}"},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        chunks = data.get("data", {}).get("chunks", [])
        if not chunks:
            return "RAGFlow 知识库中未找到相关信息。"
        lines = []
        for i, chunk in enumerate(chunks, 1):
            content = chunk.get("content", "")
            if content:
                lines.append(f"【{i}】{content}")
        result = "\n".join(lines) if lines else "RAGFlow 知识库中未找到相关信息。"
        logger.info("RAGFlow 检索成功, query='%s', 返回 %d 条结果:\n%s", query, len(lines), result[:500])
        return result
    except Exception as e:
        logger.error("RAGFlow 检索异常, query='%s': %s", query, e)
        return f"RAGFlow 知识库检索出错：{e}"
