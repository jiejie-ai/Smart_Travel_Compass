import asyncio
from types import SimpleNamespace

import pytest

from src.agent.intent import (
    CHITCHAT,
    PLAN_TRIP,
    QUICK_QUERY,
    classify_and_rewrite,
    filter_tools_for_intent,
    parse_intent,
    resolve_route,
)


def _tools(*names):
    return [SimpleNamespace(name=n) for n in names]


ALL_TOOLS = _tools(
    "do_terminate",
    "read_file",
    "write_file",
    "download_resource",
    "search_web",
    "search_knowledge_base",
    "geocode",
    "get_weather",
    "scrape_web",
    "generate_pdf",
    "search_image",
)


def _names(tools):
    return {t.name for t in tools}


# --- filter_tools_for_intent ---


def test_quick_query_excludes_pdf_and_file_tools():
    """单点问答不该拿到生成 PDF、读写文件、下载资源、终止这些重工具。"""
    result = _names(filter_tools_for_intent(ALL_TOOLS, QUICK_QUERY))
    assert "generate_pdf" not in result
    assert "read_file" not in result
    assert "write_file" not in result
    assert "download_resource" not in result
    assert "do_terminate" not in result


def test_quick_query_keeps_only_fact_lookup_tools():
    result = _names(filter_tools_for_intent(ALL_TOOLS, QUICK_QUERY))
    assert result == {
        "geocode",
        "get_weather",
        "search_web",
        "search_knowledge_base",
        "scrape_web",
    }


def test_plan_trip_gets_every_tool_untouched():
    assert _names(filter_tools_for_intent(ALL_TOOLS, PLAN_TRIP)) == _names(ALL_TOOLS)


def test_unknown_intent_fails_open_to_all_tools():
    """未知意图不能悄悄裁掉工具——回落到当前行为（全部工具）才安全。"""
    assert _names(filter_tools_for_intent(ALL_TOOLS, "something_else")) == _names(ALL_TOOLS)


def test_filter_does_not_mutate_the_input_list():
    tools = list(ALL_TOOLS)
    filter_tools_for_intent(tools, QUICK_QUERY)
    assert len(tools) == len(ALL_TOOLS)


def test_quick_query_yields_empty_set_when_no_names_match():
    """工具名全对不上时返回空集——router 靠这个信号回落到 plan_trip。"""
    assert filter_tools_for_intent(_tools("generate_pdf"), QUICK_QUERY) == []


# --- parse_intent 兜底 ---


def test_parse_intent_accepts_a_valid_quick_query():
    result = SimpleNamespace(intent="quick_query", query="天津今天天气")
    assert parse_intent(result, "天津天气咋样") == ("quick_query", "天津今天天气")


def test_parse_intent_falls_back_when_the_model_call_failed():
    assert parse_intent(None, "帮我规划天津三日游") == (PLAN_TRIP, "帮我规划天津三日游")


@pytest.mark.parametrize("bad", ["greeting", "PLAN_TRIP", "", "其它"])
def test_parse_intent_falls_back_on_out_of_schema_intent(bad):
    """实测模型在无枚举约束时会返回 'greeting' 这类值，必须拦住。"""
    result = SimpleNamespace(intent=bad, query="天津")
    assert parse_intent(result, "你好") == (PLAN_TRIP, "你好")


@pytest.mark.parametrize("blank", ["", "   ", "\n"])
def test_parse_intent_falls_back_when_rewrite_is_blank(blank):
    """改写结果为空说明这轮不可信，回落到原始输入和 plan_trip。"""
    result = SimpleNamespace(intent="quick_query", query=blank)
    assert parse_intent(result, "天津门票") == (PLAN_TRIP, "天津门票")


def test_parse_intent_keeps_chitchat_and_uses_the_original_message():
    """闲聊不做改写，下游该拿到用户原话。"""
    result = SimpleNamespace(intent="chitchat", query="")
    assert parse_intent(result, "你好") == (CHITCHAT, "你好")


def test_parse_intent_tolerates_a_result_missing_the_query_field():
    result = SimpleNamespace(intent="quick_query")
    assert parse_intent(result, "天津") == (PLAN_TRIP, "天津")


# --- resolve_route ---


def test_resolve_route_keeps_quick_query_with_its_filtered_tools():
    intent, tools = resolve_route(ALL_TOOLS, QUICK_QUERY)
    assert intent == QUICK_QUERY
    assert "generate_pdf" not in _names(tools)


def test_resolve_route_falls_back_when_quick_query_has_no_usable_tools():
    """工具名全对不上（例如 MCP 加载失败且本地工具改名）时不能进空转循环。"""
    intent, tools = resolve_route(_tools("generate_pdf", "search_image"), QUICK_QUERY)
    assert intent == PLAN_TRIP
    assert _names(tools) == {"generate_pdf", "search_image"}


def test_resolve_route_passes_plan_trip_through_untouched():
    intent, tools = resolve_route(ALL_TOOLS, PLAN_TRIP)
    assert intent == PLAN_TRIP
    assert _names(tools) == _names(ALL_TOOLS)


# --- classify_and_rewrite ---


class _FakeStructured:
    def __init__(self, result=None, exc=None):
        self._result = result
        self._exc = exc

    async def ainvoke(self, messages):
        if self._exc is not None:
            raise self._exc
        return self._result


class _FakeChatModel:
    """只记录 with_structured_output 收到的 method，并回放预设结果。"""

    def __init__(self, result=None, exc=None):
        self.method = "<not called>"
        self._structured = _FakeStructured(result, exc)

    def with_structured_output(self, schema, method=None):
        self.method = method
        return self._structured


def test_classify_requires_explicit_function_calling_method():
    """DeepSeek 上默认 method 会解析成 json_schema 并返回 400，必须显式指定。"""
    model = _FakeChatModel(SimpleNamespace(intent="quick_query", query="天津天气"))
    asyncio.run(classify_and_rewrite(model, "天津天气咋样"))
    assert model.method == "function_calling"


def test_classify_returns_the_rewritten_query_on_success():
    model = _FakeChatModel(SimpleNamespace(intent="quick_query", query="天津今天天气"))
    assert asyncio.run(classify_and_rewrite(model, "天津天气咋样")) == (
        QUICK_QUERY,
        "天津今天天气",
    )


def test_classify_falls_back_when_the_model_raises():
    model = _FakeChatModel(exc=RuntimeError("boom"))
    assert asyncio.run(classify_and_rewrite(model, "帮我规划天津三日游")) == (
        PLAN_TRIP,
        "帮我规划天津三日游",
    )
