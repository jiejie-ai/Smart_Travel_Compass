import logging
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.agent.prompts import INTENT_CLASSIFY_PROMPT

logger = logging.getLogger(__name__)

PLAN_TRIP = "plan_trip"
QUICK_QUERY = "quick_query"
CHITCHAT = "chitchat"

VALID_INTENTS = frozenset({PLAN_TRIP, QUICK_QUERY, CHITCHAT})

# 单点问答只给事实查询类工具，不带 PDF 生成、文件操作和终止工具。
QUICK_QUERY_TOOL_NAMES = frozenset({
    "geocode",
    "get_weather",
    "search_web",
    "search_knowledge_base",
    "scrape_web",
})


class QueryIntent(BaseModel):
    intent: Literal["plan_trip", "quick_query", "chitchat"] = Field(
        description="用户意图"
    )
    query: str = Field(
        description="改写后适合检索的查询；意图为 chitchat 时留空字符串"
    )


def filter_tools_for_intent(tools: list, intent: str) -> list:
    """按意图裁剪工具集。非 quick_query 一律原样返回，保持现有行为。"""
    if intent != QUICK_QUERY:
        return list(tools)
    return [tool for tool in tools if tool.name in QUICK_QUERY_TOOL_NAMES]


def resolve_route(tools: list, intent: str) -> tuple[str, list]:
    """确定最终意图及其工具集。

    quick_query 裁完一个工具都不剩（工具名对不上、MCP 加载失败等）时回落
    plan_trip，避免进入一个没有任何工具可用的空转循环。
    """
    selected = filter_tools_for_intent(tools, intent)
    if intent == QUICK_QUERY and not selected:
        logger.warning("quick_query 未匹配到任何工具，回落 plan_trip")
        return PLAN_TRIP, list(tools)
    return intent, selected


def parse_intent(result, user_message: str) -> tuple[str, str]:
    """把结构化分类结果规整成 (intent, 下游该用的消息)。

    任何不合规的情况都回落到 (plan_trip, 用户原话)，即本次改造前的行为。
    """
    if result is None:
        return PLAN_TRIP, user_message

    intent = getattr(result, "intent", None)
    if intent not in VALID_INTENTS:
        return PLAN_TRIP, user_message

    if intent == CHITCHAT:
        return CHITCHAT, user_message

    query = (getattr(result, "query", "") or "").strip()
    if not query:
        return PLAN_TRIP, user_message

    return intent, query


async def classify_and_rewrite(chat_model, user_message: str) -> tuple[str, str]:
    """一次调用同时判定意图并改写查询，避免比改造前多花一轮模型调用。

    `method="function_calling"` 必须显式指定：DeepSeek 端点不接受 langchain
    默认解析出的 json_schema 型 response_format，会直接返回 400。
    """
    try:
        structured = chat_model.with_structured_output(
            QueryIntent, method="function_calling"
        )
        result = await structured.ainvoke([
            SystemMessage(content=INTENT_CLASSIFY_PROMPT),
            HumanMessage(content=user_message),
        ])
    except Exception:
        logger.warning("意图分类调用失败，回落到 plan_trip", exc_info=True)
        return PLAN_TRIP, user_message

    return parse_intent(result, user_message)
