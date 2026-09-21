# AI旅游搭子 — Java → Python 转换设计文档

**日期**: 2026-06-26
**源项目**: `E:\000AI_tourism\ai_tourism_master` (Spring Boot 3.4.4 + Java 21)
**目标项目**: `E:\000AI_tourism_Python` (FastAPI + LangChain + LangGraph)

---

## 1. 概述

将 Java Spring Boot + Spring AI 实现的"AI旅游搭子"项目完整迁移到 Python 技术栈。

### 技术选型

| 层 | Java (原) | Python (目标) |
|---|---|---|
| Web 框架 | Spring Boot 3.4 | FastAPI |
| AI 框架 | Spring AI 1.0 | LangChain + LangGraph |
| LLM | DeepSeek (OpenAI 协议) | DeepSeek (langchain-openai) |
| Agent | 自研 ReAct Agent | LangGraph StateGraph |
| 知识库 | RAGFlow (HTTP API) | RAGFlow (HTTP API, 保留) |
| MCP 客户端 | Spring AI MCP Client | langchain-mcp-adapters |
| MCP 服务端 | Spring Boot (Java) | Python mcp 库 |
| 前端 | Vue 3 + Vite | Jinja2 + HTMX + Alpine.js |
| 会话记忆 | Kryo 文件序列化 | pickle 文件序列化 |
| PDF 生成 | iText | reportlab |
| 对象存储 | 腾讯云 COS Java SDK | cos-python-sdk-v5 |
| 包管理 | Maven | uv |
| 配置 | application.yml | pydantic-settings + .env |

### 外部依赖保留

- DeepSeek LLM（OpenAI 兼容协议）
- RAGFlow 知识库（虚拟机 Docker 部署）
- 高德地图 MCP Server（npx 启动）
- Tavily Search API
- 腾讯云 COS
- Pexels API（图片搜索 MCP Server 内部调用）

---

## 2. 项目结构

```
E:\000AI_tourism_Python\
├── pyproject.toml
├── uv.lock
├── Dockerfile
├── .env.example
│
├── src/
│   ├── __init__.py
│   ├── main.py                 # FastAPI 入口 + lifespan
│   ├── config.py               # pydantic-settings
│   │
│   ├── agent/                  # Agent 核心
│   │   ├── __init__.py
│   │   ├── state.py            # AgentState TypedDict
│   │   ├── graph.py            # LangGraph StateGraph
│   │   └── prompts.py          # System/next_step prompts
│   │
│   ├── tools/                  # 本地工具
│   │   ├── __init__.py         # get_all_tools()
│   │   ├── tavily_search.py
│   │   ├── ragflow_search.py
│   │   ├── amap_weather.py
│   │   ├── amap_geocode.py
│   │   ├── web_scraping.py
│   │   ├── pdf_generation.py
│   │   ├── file_operation.py
│   │   ├── resource_download.py
│   │   └── terminate.py
│   │
│   ├── mcp/                    # MCP 客户端
│   │   ├── __init__.py
│   │   └── client.py           # MultiServerMCPClient
│   │
│   ├── rag/                    # RAG 层
│   │   ├── __init__.py
│   │   └── query_expander.py   # 多查询扩展+去重
│   │
│   ├── memory/                 # 会话记忆
│   │   ├── __init__.py
│   │   └── file_memory.py      # FileBasedChatMemory
│   │
│   ├── api/                    # REST API
│   │   ├── __init__.py
│   │   ├── router.py           # /api/ai/travel/**
│   │   └── schemas.py          # Pydantic models
│   │
│   ├── web/                    # 前端 (Jinja2)
│   │   ├── __init__.py
│   │   ├── router.py           # 页面路由
│   │   ├── templates/
│   │   │   ├── base.html
│   │   │   ├── index.html
│   │   │   └── travel.html
│   │   └── static/
│   │       ├── style.css
│   │       └── app.js
│   │
│   └── image_search_mcp/       # 图片搜索 MCP Server
│       ├── __init__.py
│       ├── main.py             # stdio 入口
│       └── pexels_tool.py      # Pexels API
│
└── tests/
    ├── __init__.py
    ├── test_agent.py
    ├── test_tools.py
    └── test_api.py
```

---

## 3. Agent 核心层

### 3.1 AgentState

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    agent_state: str       # IDLE | RUNNING | FINISHED | ERROR
    current_step: int
    max_steps: int
    system_prompt: str
    next_step_prompt: str
```

### 3.2 StateGraph 节点图

```
START → check_state → thinker ⇄ actor → finalize → END
```

- **check_state**: IDLE→RUNNING，注入 user message + system prompt
- **thinker**: 调用 LLM (bind_tools)，有 tool_calls→actor，无→finalize
- **actor**: ToolNode 执行工具，detect terminate→finalize，否则 step+1→thinker
- **finalize**: 设置 FINISHED/ERROR，cleanup

### 3.3 条件路由

```python
def should_continue(state: AgentState) -> str:
    if state["agent_state"] in ("FINISHED", "ERROR"):
        return "finalize"
    if state["current_step"] >= state["max_steps"]:
        return "finalize"
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "actor"
    return "finalize"

def should_loop(state: AgentState) -> str:
    if state["agent_state"] in ("FINISHED", "ERROR"):
        return "finalize"
    if state["current_step"] >= state["max_steps"]:
        return "finalize"
    return "thinker"
```

### 3.4 TravelManus 配置

- `max_steps: 20`
- `system_prompt`: 角色定义 + 工具说明 + 8步工作流程 + 输出要求（原文 50 行）
- `next_step_prompt`: 每步强制调用工具、禁止短回答

### 3.5 流式事件 (SSE)

通过 `graph.astream_events(version="v2")` 获取:
- `on_chat_model_stream`: token 级文本
- `on_tool_start`: 工具开始
- `on_tool_end`: 工具完成

---

## 4. 工具层

### 4.1 本地工具（9个）

| 工具 | 函数签名 | 依赖 |
|---|---|---|
| search_web | (query: str) → str | httpx → Tavily API |
| search_knowledge_base | (query: str) → str | httpx → RAGFlow API |
| geocode | (address: str, city: str = None) → str | httpx → 高德 API |
| get_weather | (city_code: str, type: str = "base") → str | httpx → 高德 API |
| scrape_web | (url: str) → str | beautifulsoup4 |
| generate_pdf | (file_name: str, content: str) → str | reportlab + cos-python-sdk-v5 |
| read_file | (file_name: str) → str | 内置 open |
| write_file | (file_name: str, content: str) → str | 内置 open |
| do_terminate | () → str | 无 |

### 4.2 MCP 工具（2个来源）

通过 `langchain-mcp-adapters` 的 `MultiServerMCPClient` 加载:

1. **amap-maps**: `npx -y @amap/amap-maps-mcp-server` (高德地图 POI/路径规划)
2. **yu-image-search**: `python -m src.image_search_mcp.main` (Pexels 图片搜索)

### 4.3 工具合并

```python
def get_all_tools():
    local = [search_web, search_knowledge_base, geocode, get_weather,
             scrape_web, generate_pdf, read_file, write_file,
             download_resource, do_terminate]
    mcp = list(mcp_client.get_tools())
    return local + mcp
```

### 4.4 注意事项

- 工具结果截断到 3000 字符，防止 token 超限
- 每个工具有 try/except 包裹，失败返回字符串错误信息
- RAGFlow 多查询扩展：LLM 拆 query → 并行检索 → 按行去重 → 聚合

---

## 5. API 层

### 5.1 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/ai/travel/chat/sync` | TravelApp 同步聊天 |
| GET | `/api/ai/travel/chat/sse` | TravelApp SSE 流式 |
| GET | `/api/ai/travel/agent/chat` | TravelManus Agent SSE |
| GET | `/api/health` | 健康检查 |

### 5.2 TravelApp vs TravelManus

- **TravelApp**: 无工具，有文件持久化记忆，适合简单对话
- **TravelManus**: 全工具+MCP，Agent 内部 messages，适合深度规划

### 5.3 SSE 格式

- token 数据: `data: {text}\n\n`
- 工具状态: `data: [TOOL] {description}\n\n`
- 结束信号: `data: [DONE]\n\n`

---

## 6. 前端层

### 6.1 技术栈

- **服务端渲染**: FastAPI + Jinja2
- **无刷新交互**: HTMX SSE 扩展 + Alpine.js
- **Markdown 渲染**: marked.js (CDN)
- **视觉风格**: 保留珊瑚色暖色调 (#E8734A)

### 6.2 页面

| 路由 | 模板 | 说明 |
|---|---|---|
| `/` | `index.html` | 首页导航 |
| `/travel` | `travel.html` | 旅游 Agent 聊天 |

### 6.3 聊天页交互

用户输入 → EventSource 连接 SSE → 按中文标点拆分气泡（800ms 最小间隔）→ 检测 PDF URL 显示下载按钮

---

## 7. 会话记忆

```python
class FileBasedChatMemory:
    base_dir: Path  # ./tmp/chat-memory/
    # get(conversation_id) → list[Message]
    # add(conversation_id, messages) → None
    # clear(conversation_id) → None
```

- 序列化格式: pickle (替代 Kryo)
- 文件命名: `{conversation_id}.pickle`
- 仅 TravelApp 模式使用

---

## 8. 配置管理

pydantic-settings 从 `.env` 文件加载，字段映射原 `application.yml`:

- `OPENAI_API_KEY` → DeepSeek API key
- `OPENAI_BASE_URL` → `https://api.deepseek.com`
- `RAGFLOW_BASE_URL`, `RAGFLOW_API_KEY`, `RAGFLOW_KB_ID`
- `AMAP_API_KEY`
- `TAVILY_API_KEY`
- `PEXELS_API_KEY`
- `COS_SECRET_ID`, `COS_SECRET_KEY`, `COS_REGION`, `COS_BUCKET_NAME`

---

## 9. 图片搜索 MCP Server (Python 重写)

原 Java `yu-image-search-mcp-server` 改为 Python 实现:

```python
# src/image_search_mcp/main.py
import mcp.server.stdio
from mcp.server import Server

server = Server("yu-image-search")

@server.tool()
async def search_image(query: str) -> str:
    # 调用 Pexels API，返回 medium 尺寸图片 URL 列表
    ...

if __name__ == "__main__":
    mcp.server.stdio.run(server)
```

---

## 10. 部署

### Dockerfile
```dockerfile
FROM python:3.12-slim
WORKDIR /app
RUN pip install uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY src/ src/
EXPOSE 8123
CMD ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8123"]
```

### 外部依赖（不动）
- RAGFlow: 虚拟机 Docker Compose
- PostgreSQL: Docker Desktop
- Node.js (npx): 高德 MCP Server 运行时自动拉取

---

## 11. 与原始项目的映射

| Java | Python |
|---|---|
| `BaseAgent.java` | `agent/graph.py` (LangGraph StateGraph) |
| `ReActAgent.java` | `agent/graph.py` (thinker + actor 节点) |
| `ToolCallAgent.java` | `agent/graph.py` (ToolNode + 条件路由) |
| `TravelManus.java` | `agent/prompts.py` + `agent/graph.py` (配置) |
| `AgentState.java` | `agent/state.py` (TypedDict) |
| `ToolRegistration.java` | `tools/__init__.py` (get_all_tools) |
| `AiController.java` | `api/router.py` |
| `TravelApp.java` | `api/router.py` (chat/sync, chat/sse) |
| `FileBasedChatMemory.java` | `memory/file_memory.py` |
| `MyLoggerAdvisor.java` | LangGraph built-in logging |
| `RagflowQueryExpander` | `rag/query_expander.py` |
| `PDFGenerationTool.java` | `tools/pdf_generation.py` |
| `ImageSearchTool.java` (MCP) | `image_search_mcp/pexels_tool.py` |
| `YuAiAgentApplication.java` | `main.py` |
| `application.yml` | `.env` + `config.py` |
| Vue 3 前端 | `web/templates/` + `web/static/` |
| `mcp-servers.json` | `mcp/client.py` (代码内配置) |
| `pom.xml` | `pyproject.toml` |
