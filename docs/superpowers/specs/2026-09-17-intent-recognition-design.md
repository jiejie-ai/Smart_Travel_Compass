# 意图识别设计文档

**日期**: 2026-09-17
**项目**: AI 旅游搭子 (`E:\000AI_tourism_Python`)
**改动范围**: `src/agent/` — 在请求主链路上插入意图分类层

---

## 1. 背景与问题

当前 `travel_agent_chat` 端点把所有用户输入无差别推进同一条 ReAct 循环,而 `TRAVEL_MANUS_SYSTEM_PROMPT` 把输出要求写死:

```
OUTPUT REQUIREMENTS:
Your final answer MUST be a COMPREHENSIVE travel guide (at least 700 Chinese characters) covering:
...
After your text answer, call generatePDF with the FULL guide content and image URLs, then call doTerminate.
```

后果:用户问"天津今天天气怎么样"、甚至只说一句"你好",也会被要求产出 700 字攻略并生成 PDF。系统没有"这个问题不该走重流程"的判断点。

同时,意图判断目前是**隐式**的——藏在工具选择里(`thinker` 节点让模型自选工具),不可观测、不可控制。

### 目标

1. 区分三类意图,让轻量请求不再被重流程污染
2. 意图成为显式、可观测的状态
3. 不改变 `plan_trip` 的现有行为
4. **最坏情况等于没改**——分类失败时退回当前行为

### 非目标

- 不做意图的循环内修正(分类发生在进图之前,进图后不再变)
- 不做多轮行程修改(需要可寻址的行程状态,当前记忆是纯文本历史)
- 不建分类质量评估集(见第 8 节)

---

## 2. 意图分类

| 意图 | 含义 | 走法 |
|---|---|---|
| `plan_trip` | 要完整攻略 + PDF | 现有完整图,**一字不改** |
| `quick_query` | 单点事实:天气、门票、交通、某地信息 | 同一张图,换提示词 + 裁工具 |
| `chitchat` | 打招呼、问能力、与旅行无关 | 不进图,单次流式调用 |

### 判据

- `plan_trip`:要求行程规划、攻略、多维度综合建议
- `quick_query`:单个事实性问题,一句话能答完,不需要 PDF
- `chitchat`:无信息需求,或与旅行无关

---

## 3. 分类的产生方式

### 3.1 与查询改写合并为一次调用

`src/api/router.py:73-82` 现在已有一次额外的模型调用做查询改写。把这轮扩展为同时输出意图和改写后的查询,**净增延迟为零**。

```python
# src/agent/intent.py — 新文件
class QueryIntent(BaseModel):
    intent: Literal["plan_trip", "quick_query", "chitchat"]
    query: str


async def classify_and_rewrite(chat_model, user_message: str) -> tuple[str, str]:
    """失败一律兜底为 ("plan_trip", 原文)。"""
```

### 3.2 结构化输出:必须显式指定 function_calling

**探针已验证(2026-09-17),这是本设计最容易踩的坑:**

| method | 结果 |
|---|---|
| 默认(不传) | **失败**。langchain 解析为 `json_schema` 型 response_format,DeepSeek 返回 400 `This response_format type is unavailable now` |
| `json_mode` | 可达但**不可用**。无枚举约束,模型自由发挥——实测把 `chitchat` 输出成 `greeting`,`Literal` 校验直接失败 |
| `"function_calling"` | **可用**。schema 作为工具定义下发,枚举被约束,`Literal` 校验通过 |

因此必须写死:

```python
structured = chat_model.with_structured_output(QueryIntent, method="function_calling")
```

不传 `method` 会在 DeepSeek 上直接 400。

### 3.3 兜底(本设计最重要的安全属性)

以下任一情况,一律回落为 `("plan_trip", 原文)`:

- 模型调用抛异常
- 结构化解析失败 / Pydantic 校验失败
- 返回的 `query` 为空且意图不是 `chitchat`

即**最坏情况等于本次改造没做**。这保证了这次改动不会让线上行为比现在更差。

---

## 4. 三条路径

| 意图 | 工具集 | max_steps | 结束方式 |
|---|---|---|---|
| `plan_trip` | 全部 | 20 | `do_terminate` 或自然收尾 |
| `quick_query` | `geocode` / `get_weather` / `search_web` / `search_knowledge_base` / `scrape_web` | 6 | 只能自然收尾 |
| `chitchat` | 无 | — | 单次调用直接结束 |

### 4.1 工具裁剪为什么便宜

`bind_tools` 和 `ToolNode` 都是**编译期**固定的,运行期改不了。但 `src/api/router.py:84` 已经是每请求重建图:

```python
agent_graph = build_travel_manus_graph(chat_model, tools)
```

所以按意图裁工具只需给这次构建传子集,**`graph.py` 的控制流一行不用改**,也不需维护多张图。

### 4.2 quick_query 为什么不需要 do_terminate

`_should_continue` 已经有一条自然收尾路径(`src/agent/graph.py:34-44`):

```python
if hasattr(last, "tool_calls") and last.tool_calls:
    return "actor"
return "finalize"
```

模型给出最终文本回答时不带 `tool_calls`,这条边就走 `finalize`。所以裁掉 `do_terminate` 后循环仍能正常结束,`_should_loop` 的终止判定逻辑不受影响。

### 4.3 chitchat 不复用 TRAVEL_APP_SYSTEM_PROMPT

`TRAVEL_APP_SYSTEM_PROMPT` 已被 `travel_chat_sync`(`router.py:32`)和 `travel_chat_sse`(`router.py:50`)两个端点使用。复用它做闲聊会让三个端点隐式耦合,改一处动三处。改为新增独立的 `CHITCHAT_SYSTEM_PROMPT`。

---

## 5. 状态变更

`AgentState` 增加第 7 个字段:

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    agent_state: str
    current_step: int
    max_steps: int
    system_prompt: str
    next_step_prompt: str
    intent: str  # plan_trip | quick_query | chitchat
```

`build_initial_state(user_message, intent="plan_trip")` —— **默认值即当前行为**,调用方不传参保持原样。

`build_initial_state` 按意图选择 `system_prompt`、`next_step_prompt`、`max_steps`。

**`graph.py` 的控制流(节点、边、两个条件函数)不做任何修改。**

---

## 6. 提示词拆分

`TRAVEL_MANUS_SYSTEM_PROMPT` 现在是一坨:角色 + 工具清单 + 工作流 + 700 字要求 + PDF/终止指令。三个意图共用它是问题的根源,必须拆。

### 6.1 新增

| 常量 | 用途 |
|---|---|
| `INTENT_CLASSIFY_PROMPT` | 分类 + 改写的系统提示词 |
| `QUICK_QUERY_SYSTEM_PROMPT` | 单点问答:简短直接回答,不出 PDF,不写 700 字 |
| `QUICK_QUERY_NEXT_STEP_PROMPT` | 简短续跑提示,不含 PDF/终止要求 |
| `CHITCHAT_SYSTEM_PROMPT` | 自我介绍 + 反问目的地/时间/预算/偏好,结尾引导到规划 |

### 6.2 顺手修复:提示词里的工具名是错的

现状(`src/agent/prompts.py:4-12`)提示词用驼峰名,与实际注册名不一致:

| 提示词里写的 | 实际注册名 |
|---|---|
| `searchWeb` | `search_web` |
| `searchKnowledgeBase` | `search_knowledge_base` |
| `generatePDF` | `generate_pdf` |
| `doTerminate` | `do_terminate` |
| `searchImage` | MCP 工具,名称以启动日志为准 |
| `getWeather` | `get_weather` |
| `scrapeWeb` | `scrape_web` |
| **`fileOperation`** | **不存在**。实际是 `read_file` / `write_file` |

反正要重写这三份提示词,一并改掉。

### 6.3 一致性测试守住

新增测试:**断言每份 system prompt 中出现的工具名,都真实存在于该类意图会绑定的工具集里**。

这个测试如果早写了,上述工具名 bug 根本进不去。它同时守住未来"改了提示词忘了改工具注册"的漂移。

### 6.4 一致性测试的可判定性:工具名清单从哪来

代理可见的工具分三类:

| 来源 | 数量 | 名称可否静态获知 |
|---|---|---|
| 本地工具 `get_local_tools()` | 10 | **可以**,直接读函数名 |
| `yu-image-search` MCP | 1,`search_image` | **可以**,源码里 `@mcp.tool` 装饰的函数名(`src/image_search_mcp/main.py:41`) |
| `amap-maps` MCP | 若干 | **不可以**。名称由第三方 npm 包 `@amap/amap-maps-mcp-server` 在运行期提供,本项目文档中无记录 |

因此单测无法枚举完整工具集。处理方式:

1. 在 `src/tools/__init__.py` 里声明静态可判定的名称集合:

```python
LOCAL_TOOL_NAMES = frozenset(t.name for t in get_local_tools())
EXTERNAL_TOOL_NAMES = frozenset({"search_image"})  # 与 MCP server 源码同步维护
```

2. 一致性测试断言:**提示词里出现的每个工具名,都必须属于 `LOCAL_TOOL_NAMES | EXTERNAL_TOOL_NAMES`**。
3. **重写后的提示词不引用 `amap-maps` 的任何工具**——它们与本地 `geocode` / `get_weather` 功能重叠,提示词里指向本地版本即可。这样一致性测试完全静态可判定,不依赖启动 MCP 子进程。
4. `EXTERNAL_TOOL_NAMES` 需要人工同步。若 MCP server 改名而此处未更新,一致性测试会失败——这是**期望行为**(宁可测试失败也不要提示词静默指向不存在的工具)。

---

## 7. 接入 router

### 7.1 分支结构

```python
# src/api/router.py — 概念示意
intent, rewritten = await classify_and_rewrite(chat_model, message)

if intent == "chitchat":
    return StreamingResponse(chitchat_stream(chat_model, rewritten))

tools = filter_tools_for_intent(tools, intent)
graph = build_travel_manus_graph(chat_model, tools)
initial_state = build_initial_state(rewritten, intent=intent)
```

### 7.2 SSE 协议

新增一个事件前缀,置于 `[REWRITE]` 之前:

```
data: [INTENT] plan_trip
data: [REWRITE] <改写后的查询>
...
```

现有前缀 `[REWRITE]` / `[TOOL]` / `[DONE]` / `[ERROR]` 不变。

**前端必须同步处理**:`src/web/templates/travel.html` 的 SSE 解析器若不认 `[INTENT]`,会把这行原始文本渲染到界面上。

### 7.3 防御性检查

若 `filter_tools_for_intent` 对 `quick_query` 返回空集(例如 MCP 加载失败且工具名变更),回落为 `plan_trip`,不进入一个没有工具可用的循环。

---

## 8. 测试策略

### 8.1 单测覆盖(确定性边界)

| # | 测试对象 | 断言 |
|---|---|---|
| 1 | `filter_tools_for_intent` | `quick_query` 排除 `generate_pdf` / `read_file` / `write_file` / `download_resource` / `do_terminate` |
| 2 | `parse_intent` 兜底 | 畸形 JSON / 未知意图串 / 空串 / 校验失败,一律回落 `plan_trip` |
| 3 | `build_initial_state(msg, intent)` | 字段正确,且按意图选中对应提示词与 `max_steps` |
| 4 | 提示词与工具集一致性 | 提示词里的工具名都真实存在(见 6.3) |
| 5 | `quick_query` 路径 | `do_terminate` 不在工具集内,且循环仍能 `finalize` |
| 6 | 默认参数 | `build_initial_state(msg)` 不传 intent 时行为与改造前一致 |

现有 `tests/test_agent.py` 的 `_state()` 辅助函数需要补 `intent` 字段。

### 8.2 测不到的:分类质量

**单元测试无法覆盖分类准确率。** 探针用 8 条话术粗测 8/8 正确,但这不是评估。

要真正验证需要一批带标签的真实话术(建议 30–50 条,覆盖三类、含边界样本如"帮我看看天津天气然后规划一下行程"),跑准确率和混淆矩阵。

**本项目当前没有评估集,本次不做**,作为已知缺口记录。这意味着意图分类上线后如果分错,只能靠人工发现。

---

## 9. 涉及文件

| 文件 | 改动 |
|---|---|
| `src/agent/intent.py` | **新增**。`QueryIntent`、`classify_and_rewrite`、`filter_tools_for_intent`、`parse_intent` 兜底逻辑 |
| `src/agent/state.py` | 加 `intent` 字段 |
| `src/agent/prompts.py` | 新增 4 个提示词常量;修工具名 bug |
| `src/agent/graph.py` | 只改 `build_initial_state`。**控制流不动** |
| `src/api/router.py` | 合并调用、三条分支、`[INTENT]` SSE 事件 |
| `src/web/templates/travel.html` | SSE 解析器认 `[INTENT]` |
| `tests/test_intent.py` | **新增** |
| `tests/test_agent.py` | `_state()` 补 `intent` 字段 |

---

## 10. 风险

| 风险 | 缓解 |
|---|---|
| 分类错误导致重请求走轻路径(用户要攻略却只得到一句话) | 提示词优先判 `plan_trip`;兜底默认 `plan_trip`;边界样本纳入日后评估集 |
| 合并调用后,改写质量受分类任务挤占 | 探针实测改写结果正常;失败仍有 `原文` 兜底 |
| DeepSeek 端点行为变更 | `method="function_calling"` 显式写死,并有单测守住 parse 兜底 |
| 前端漏处理 `[INTENT]` 导致界面出现原始文本 | 实现时同步改 `travel.html`,并在浏览器里实测 |

---

## 11. 交付判定

- [ ] `tests/` 全绿,新增测试覆盖 8.1 全部六项
- [ ] `plan_trip` 路径行为与改造前完全一致
- [ ] 闲聊请求不再生成 PDF
- [ ] 单点问答请求能在 6 步内结束且不出 PDF
- [ ] 浏览器实测三条路径,SSE 无原始前缀泄漏到界面
