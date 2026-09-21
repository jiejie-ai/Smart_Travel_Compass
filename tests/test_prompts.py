import re

import pytest

from src.agent import prompts
from src.tools import EXTERNAL_TOOL_NAMES, LOCAL_TOOL_NAMES

# "可用工具"段落里每条以 `- <工具名>: 说明` 的形式列出。
TOOL_BULLET = re.compile(r"^-\s+([A-Za-z_][A-Za-z0-9_]*)\s*[:：]", re.M)

REAL_TOOL_NAMES = LOCAL_TOOL_NAMES | EXTERNAL_TOOL_NAMES


def _mentioned_tool_names(prompt: str) -> set[str]:
    return set(TOOL_BULLET.findall(prompt))


def test_the_parser_actually_finds_the_listed_tools():
    """先证明扫描器有效，否则下面的测试会因扫不到东西而空过。"""
    assert _mentioned_tool_names("- geocode: 把城市名转成坐标\n") == {"geocode"}


@pytest.mark.parametrize(
    "prompt_name",
    [
        "TRAVEL_MANUS_SYSTEM_PROMPT",
        "QUICK_QUERY_SYSTEM_PROMPT",
    ],
)
def test_every_tool_listed_in_a_prompt_actually_exists(prompt_name):
    """提示词里列出的工具必须真实注册过。

    历史上这里写的是驼峰名 searchWeb/searchKnowledgeBase/generatePDF/doTerminate，
    以及一个根本不存在的 fileOperation，模型照着调只会一直失败。
    """
    prompt = getattr(prompts, prompt_name)
    ghost_tools = _mentioned_tool_names(prompt) - REAL_TOOL_NAMES
    assert ghost_tools == set(), f"{prompt_name} 引用了不存在的工具: {ghost_tools}"


def test_the_prompt_scanner_is_not_a_no_op_on_the_real_prompts():
    """确认两份提示词确实列出了工具，否则上一条测试是假绿。"""
    for prompt_name in ("TRAVEL_MANUS_SYSTEM_PROMPT", "QUICK_QUERY_SYSTEM_PROMPT"):
        assert _mentioned_tool_names(getattr(prompts, prompt_name)), prompt_name


def test_plan_trip_prompt_keeps_the_full_guide_contract():
    prompt = prompts.TRAVEL_MANUS_SYSTEM_PROMPT
    assert "700" in prompt
    assert "generate_pdf" in prompt


def test_quick_query_prompt_does_not_demand_a_full_guide_or_a_pdf():
    """单点问答的痛点就是被 700 字攻略 + PDF 污染，这里守住。"""
    prompt = prompts.QUICK_QUERY_SYSTEM_PROMPT
    assert "700" not in prompt
    assert "generate_pdf" not in prompt


def test_quick_query_prompt_forbids_the_tools_it_will_not_be_given():
    prompt = prompts.QUICK_QUERY_SYSTEM_PROMPT
    for name in ("read_file", "write_file", "download_resource", "do_terminate"):
        assert name not in prompt


def test_chitchat_prompt_is_independent_from_the_shared_app_prompt():
    """闲聊不能复用 TRAVEL_APP_SYSTEM_PROMPT——那会让三个端点隐式耦合。"""
    assert prompts.CHITCHAT_SYSTEM_PROMPT != prompts.TRAVEL_APP_SYSTEM_PROMPT
    assert prompts.CHITCHAT_SYSTEM_PROMPT.strip()


def test_quick_query_next_step_prompt_does_not_mention_pdf_or_terminate():
    prompt = prompts.QUICK_QUERY_NEXT_STEP_PROMPT
    assert "generate_pdf" not in prompt
    assert "do_terminate" not in prompt
