import json
import logging

from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.graph import build_initial_state, build_travel_manus_graph
from src.agent.intent import CHITCHAT, classify_and_rewrite, resolve_route
from src.agent.prompts import CHITCHAT_SYSTEM_PROMPT, TRAVEL_APP_SYSTEM_PROMPT
from src.memory.file_memory import FileBasedChatMessageHistory

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/ai")

MEMORY_DIR = "tmp/chat-memory"


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.get("/travel/chat/sync")
async def travel_chat_sync(
    request: Request,
    message: str = Query(...),
    chat_id: str = Query("default"),
):
    chat_model = request.app.state.chat_model
    memory = FileBasedChatMessageHistory(MEMORY_DIR, chat_id)
    history_messages = list(memory.messages)
    msg_list = [SystemMessage(content=TRAVEL_APP_SYSTEM_PROMPT)] + history_messages
    msg_list.append(HumanMessage(content=message))

    response = chat_model.invoke(msg_list)
    memory.add_messages([HumanMessage(content=message), response])
    return {"content": response.content}


@router.get("/travel/chat/sse")
async def travel_chat_sse(
    request: Request,
    message: str = Query(...),
    chat_id: str = Query("default"),
):
    chat_model = request.app.state.chat_model

    async def event_stream():
        messages = [
            SystemMessage(content=TRAVEL_APP_SYSTEM_PROMPT),
            HumanMessage(content=message),
        ]
        try:
            async for chunk in chat_model.astream(messages):
                if chunk.content:
                    yield f"data: {json.dumps(chunk.content, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error("SSE error: %s", e)
            yield f"data: [ERROR] {e}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


async def _chitchat_stream(chat_model, message: str):
    """闲聊不进 agent 图：没有工具、没有 ReAct 循环、不生成 PDF。"""
    try:
        yield f"data: [INTENT] {CHITCHAT}\n\n"
        messages = [
            SystemMessage(content=CHITCHAT_SYSTEM_PROMPT),
            HumanMessage(content=message),
        ]
        async for chunk in chat_model.astream(messages):
            if chunk.content:
                yield f"data: {json.dumps(chunk.content, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error("Chitchat SSE error: %s", e)
        yield f"data: [ERROR] {e}\n\n"


@router.get("/travel/agent/chat")
async def travel_agent_chat(
    request: Request,
    message: str = Query(...),
):
    chat_model = request.app.state.chat_model
    tools = request.app.state.tools

    # 一次调用同时判定意图并改写查询，不比改造前多花一轮模型调用
    intent, resolved = await classify_and_rewrite(chat_model, message)
    logger.info("意图: %s | 下游消息: '%s'", intent, resolved)

    if intent == CHITCHAT:
        return StreamingResponse(
            _chitchat_stream(chat_model, resolved),
            media_type="text/event-stream",
        )

    intent, tools = resolve_route(tools, intent)
    agent_graph = build_travel_manus_graph(chat_model, tools)
    initial_state = build_initial_state(resolved, intent=intent)

    async def event_stream():
        try:
            yield f"data: [INTENT] {intent}\n\n"
            # 改写结果来自模型，用 json 编码避免其中的换行破坏 SSE 帧
            yield f"data: [REWRITE] {json.dumps(resolved, ensure_ascii=False)}\n\n"
            async for event in agent_graph.astream_events(initial_state, version="v2", config={"recursion_limit": 100}):
                kind = event.get("event", "")
                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if chunk.content:
                        yield f"data: {json.dumps(chunk.content, ensure_ascii=False)}\n\n"
                elif kind == "on_tool_start":
                    name = event.get("name", "unknown")
                    yield f"data: [TOOL] 正在调用 {name}...\n\n"
                elif kind == "on_tool_end":
                    name = event.get("name", "unknown")
                    yield f"data: [TOOL] {name} 完成\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error("Agent SSE error: %s", e)
            yield f"data: [ERROR] {e}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
