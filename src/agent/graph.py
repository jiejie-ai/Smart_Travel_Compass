import logging

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from src.agent.state import AgentState
from src.agent.intent import CHITCHAT, PLAN_TRIP, QUICK_QUERY
from src.agent.prompts import (
    CHITCHAT_SYSTEM_PROMPT,
    QUICK_QUERY_NEXT_STEP_PROMPT,
    QUICK_QUERY_SYSTEM_PROMPT,
    TRAVEL_MANUS_SYSTEM_PROMPT,
    TRAVEL_MANUS_NEXT_STEP_PROMPT,
)
from src.tools.terminate import do_terminate

logger = logging.getLogger(__name__)

# 意图 -> (system_prompt, next_step_prompt, max_steps)
# 单点问答给更小的步数预算：它本来就该几步内答完，超了说明走歪了。
INTENT_PROFILES = {
    PLAN_TRIP: (TRAVEL_MANUS_SYSTEM_PROMPT, TRAVEL_MANUS_NEXT_STEP_PROMPT, 20),
    QUICK_QUERY: (QUICK_QUERY_SYSTEM_PROMPT, QUICK_QUERY_NEXT_STEP_PROMPT, 6),
    CHITCHAT: (CHITCHAT_SYSTEM_PROMPT, "", 1),
}


def _build_llm_with_tools(chat_model, tools):
    return chat_model.bind_tools(tools)


def _make_thinker(llm_with_tools):
    def thinker(state: AgentState) -> dict:
        messages = list(state["messages"])
        if state["current_step"] > 0 and state.get("next_step_prompt"):
            messages.append(HumanMessage(content=state["next_step_prompt"]))
        response = llm_with_tools.invoke(messages)
        return {
            "messages": [response],
            "current_step": state["current_step"] + 1,
        }
    return thinker


def _should_continue(state: AgentState) -> str:
    if state.get("agent_state") in ("FINISHED", "ERROR"):
        return "finalize"
    if state.get("current_step", 0) >= state.get("max_steps", 20):
        return "finalize"
    messages = state["messages"]
    if messages:
        last = messages[-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "actor"
    return "finalize"


def _terminate_was_called(messages: list) -> bool:
    """扫本轮工具结果，判断终止工具是否被执行过。

    只看本轮（从尾部回扫到上一条 AI 消息为止），且按工具名匹配而非返回值内容，
    避免结果排列顺序或返回文案影响终止判定。
    """
    for message in reversed(messages):
        if not isinstance(message, ToolMessage):
            break
        if message.name == do_terminate.name:
            return True
    return False


def _should_loop(state: AgentState) -> str:
    if state.get("agent_state") in ("FINISHED", "ERROR"):
        return "finalize"
    if state.get("current_step", 0) >= state.get("max_steps", 20):
        return "finalize"
    if _terminate_was_called(state.get("messages", [])):
        return "finalize"
    return "thinker"


def _check_state(state: AgentState) -> dict:
    if state.get("agent_state") != "RUNNING":
        return {"agent_state": "RUNNING"}
    return {}


def _finalize(state: AgentState) -> dict:
    return {"agent_state": "FINISHED"}


def build_travel_manus_graph(chat_model, tools):
    llm_with_tools = _build_llm_with_tools(chat_model, tools)
    thinker = _make_thinker(llm_with_tools)

    builder = StateGraph(AgentState)
    builder.add_node("check_state", _check_state)
    builder.add_node("thinker", thinker)
    builder.add_node("actor", ToolNode(tools))
    builder.add_node("finalize", _finalize)

    builder.add_edge(START, "check_state")
    builder.add_edge("check_state", "thinker")
    builder.add_conditional_edges("thinker", _should_continue, {
        "actor": "actor",
        "finalize": "finalize",
    })
    builder.add_conditional_edges("actor", _should_loop, {
        "thinker": "thinker",
        "finalize": "finalize",
    })
    builder.add_edge("finalize", END)

    return builder.compile()


def build_initial_state(user_message: str, intent: str = PLAN_TRIP) -> AgentState:
    if intent not in INTENT_PROFILES:
        intent = PLAN_TRIP
    system_prompt, next_step_prompt, max_steps = INTENT_PROFILES[intent]

    return AgentState(
        messages=[
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ],
        agent_state="RUNNING",
        current_step=0,
        max_steps=max_steps,
        system_prompt=system_prompt,
        next_step_prompt=next_step_prompt,
        intent=intent,
    )
