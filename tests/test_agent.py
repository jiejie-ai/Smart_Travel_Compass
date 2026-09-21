from langchain_core.messages import AIMessage, ToolMessage

from src.agent import prompts
from src.agent.graph import _should_loop, build_initial_state
from src.agent.state import AgentState


def test_agent_state_defaults():
    state = AgentState(
        messages=[],
        agent_state="IDLE",
        current_step=0,
        max_steps=20,
        system_prompt="test",
        next_step_prompt="test",
        intent="plan_trip",
    )
    assert state["agent_state"] == "IDLE"
    assert state["max_steps"] == 20
    assert state["current_step"] == 0


def _state(messages, current_step=1, max_steps=20, agent_state="RUNNING"):
    return AgentState(
        messages=messages,
        agent_state=agent_state,
        current_step=current_step,
        max_steps=max_steps,
        system_prompt="test",
        next_step_prompt="test",
        intent="plan_trip",
    )


# --- build_initial_state 按意图装配 ---


def test_build_initial_state_defaults_to_the_existing_plan_trip_behaviour():
    """不传 intent 时必须与改造前完全一致，老调用方不受影响。"""
    state = build_initial_state("帮我规划天津三日游")
    assert state["intent"] == "plan_trip"
    assert state["system_prompt"] == prompts.TRAVEL_MANUS_SYSTEM_PROMPT
    assert state["next_step_prompt"] == prompts.TRAVEL_MANUS_NEXT_STEP_PROMPT
    assert state["max_steps"] == 20


def test_build_initial_state_quick_query_swaps_prompts_and_lowers_the_step_budget():
    state = build_initial_state("天津今天天气", intent="quick_query")
    assert state["intent"] == "quick_query"
    assert state["system_prompt"] == prompts.QUICK_QUERY_SYSTEM_PROMPT
    assert state["next_step_prompt"] == prompts.QUICK_QUERY_NEXT_STEP_PROMPT
    assert state["max_steps"] == 6


def test_build_initial_state_chitchat_selects_the_chitchat_prompt():
    state = build_initial_state("你好", intent="chitchat")
    assert state["intent"] == "chitchat"
    assert state["system_prompt"] == prompts.CHITCHAT_SYSTEM_PROMPT


def test_build_initial_state_normalises_an_unknown_intent_to_plan_trip():
    state = build_initial_state("x", intent="nonsense")
    assert state["intent"] == "plan_trip"
    assert state["max_steps"] == 20


def test_build_initial_state_puts_the_chosen_prompt_first_and_the_user_last():
    state = build_initial_state("天津")
    assert state["messages"][0].content == prompts.TRAVEL_MANUS_SYSTEM_PROMPT
    assert state["messages"][-1].content == "天津"


def _tool_turn(*tool_names):
    """One assistant turn requesting several tools, plus their results."""
    calls = [
        {"name": name, "args": {}, "id": f"call_{i}"}
        for i, name in enumerate(tool_names)
    ]
    return [AIMessage(content="", tool_calls=calls)]


def _tool_result(name, index, content="ok"):
    return ToolMessage(content=content, name=name, tool_call_id=f"call_{index}")


def test_should_loop_finalizes_when_terminate_result_is_not_last():
    """一轮内同时调用终止工具和其他工具时，不应依赖结果的排列顺序。"""
    state = _state(
        _tool_turn("do_terminate", "generate_pdf")
        + [_tool_result("do_terminate", 0, "任务结束")]
        + [_tool_result("generate_pdf", 1, "PDF 已生成")]
    )
    assert _should_loop(state) == "finalize"


def test_should_loop_finalizes_regardless_of_terminate_tool_output_text():
    """终止判定不应依赖工具返回的文案。"""
    state = _state(
        _tool_turn("do_terminate") + [_tool_result("do_terminate", 0, "done")]
    )
    assert _should_loop(state) == "finalize"


def test_should_loop_continues_when_tool_content_merely_mentions_ending():
    """普通工具返回的正文里出现"任务结束"，不代表终止工具被调用过。"""
    state = _state(
        _tool_turn("search_knowledge_base")
        + [
            _tool_result(
                "search_knowledge_base",
                0,
                "琉璃胡同半日游到此任务结束，可顺路去王府井。",
            )
        ]
    )
    assert _should_loop(state) == "thinker"


def test_should_loop_continues_after_a_normal_tool_result():
    state = _state(
        _tool_turn("get_weather") + [_tool_result("get_weather", 0, '{"weather":"晴"}')]
    )
    assert _should_loop(state) == "thinker"


def test_should_loop_finalizes_when_max_steps_reached():
    state = _state(
        _tool_turn("get_weather") + [_tool_result("get_weather", 0)],
        current_step=20,
        max_steps=20,
    )
    assert _should_loop(state) == "finalize"


def test_should_loop_finalizes_when_state_already_terminal():
    state = _state(
        _tool_turn("get_weather") + [_tool_result("get_weather", 0)],
        agent_state="FINISHED",
    )
    assert _should_loop(state) == "finalize"
