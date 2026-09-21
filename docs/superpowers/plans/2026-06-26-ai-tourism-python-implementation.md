# AI Tourism Python — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert Java Spring Boot "AI旅游搭子" to Python (FastAPI + LangChain + LangGraph), preserving all functionality.

**Architecture:** LangGraph StateGraph for the ReAct agent loop, LangChain @tool for 9 local tools, langchain-mcp-adapters for 2 MCP servers, FastAPI SSE for streaming, Jinja2+Alpine.js for frontend.

**Tech Stack:** Python 3.12+, FastAPI, LangChain 0.3+, LangGraph 0.2+, langchain-openai, httpx, beautifulsoup4, reportlab, cos-python-sdk-v5, Jinja2, Alpine.js, marked.js

## Global Constraints

- Python >= 3.12
- All external services preserved (DeepSeek, RAGFlow, Tavily, Amap, COS, Pexels)
- Package manager: uv
- No Node.js dependency (frontend uses Jinja2 + Alpine.js CDN)
- Original Java system prompts copied verbatim
- Tool result truncation at 3000 chars
- PDF uses reportlab with Chinese font support

---

### Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `src/__init__.py`
- Create: `src/config.py`

**Interfaces:**
- Produces: `Settings` class (pydantic-settings), `pyproject.toml` with all deps

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "ai-tourism"
version = "0.1.0"
description = "AI Tourism Assistant - AI旅游搭子"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.34",
    "langchain>=0.3",
    "langgraph>=0.2",
    "langchain-openai>=0.3",
    "langchain-mcp-adapters>=0.1",
    "httpx>=0.28",
    "beautifulsoup4>=4.12",
    "reportlab>=4.2",
    "cos-python-sdk-v5>=1.9",
    "jinja2>=3.1",
    "pydantic-settings>=2.7",
    "python-multipart>=0.0.18",
    "mcp>=1.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

- [ ] **Step 2: Create .env.example**

```
OPENAI_API_KEY=sk-your-deepseek-key
OPENAI_BASE_URL=https://api.deepseek.com
CHAT_MODEL=deepseek-chat
TEMPERATURE=0.7
RAGFLOW_BASE_URL=http://192.168.138.137:81
RAGFLOW_API_KEY=your-ragflow-key
RAGFLOW_KB_ID=your-kb-id
AMAP_API_KEY=your-amap-key
TAVILY_API_KEY=tvly-your-tavily-key
PEXELS_API_KEY=your-pexels-key
COS_SECRET_ID=your-cos-secret-id
COS_SECRET_KEY=your-cos-secret-key
COS_REGION=ap-guangzhou
COS_BUCKET_NAME=your-bucket
SERVER_PORT=8123
```

- [ ] **Step 3: Create src/config.py**

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    openai_api_key: str = "changeme"
    openai_base_url: str = "https://api.deepseek.com"
    chat_model: str = "deepseek-chat"
    temperature: float = 0.7

    ragflow_base_url: str = "http://192.168.138.137:81"
    ragflow_api_key: str = "changeme"
    ragflow_kb_id: str = "changeme"

    amap_api_key: str = "changeme"
    tavily_api_key: str = "changeme"
    pexels_api_key: str = "changeme"

    cos_secret_id: str = "changeme"
    cos_secret_key: str = "changeme"
    cos_region: str = "ap-guangzhou"
    cos_bucket_name: str = "changeme"

    server_port: int = 8123
    server_host: str = "0.0.0.0"

settings = Settings()
```

- [ ] **Step 4: Create src/__init__.py** — empty file

- [ ] **Step 5: Install dependencies**

Run: `cd E:/000AI_tourism_Python && uv sync`
Expected: all packages installed without error

---

### Task 2: Agent State + Prompts

**Files:**
- Create: `src/agent/__init__.py`
- Create: `src/agent/state.py`
- Create: `src/agent/prompts.py`

**Interfaces:**
- Produces: `AgentState` TypedDict, `TRAVEL_MANUS_SYSTEM_PROMPT`, `TRAVEL_MANUS_NEXT_STEP_PROMPT`, `TRAVEL_APP_SYSTEM_PROMPT`

- [ ] **Step 1: Create src/agent/state.py**

```python
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    agent_state: str
    current_step: int
    max_steps: int
    system_prompt: str
    next_step_prompt: str
```

- [ ] **Step 2: Create src/agent/prompts.py** — exact copy of Java prompts

```python
TRAVEL_MANUS_SYSTEM_PROMPT = """\
You are TravelManus, an AI travel assistant (AI旅游搭子). You have access to REAL tools and MUST use them to gather information before answering. NEVER answer from your own knowledge — always search first.

Available tools:
- searchWeb: Search the internet for latest travel guides, Xiaohongshu tips, attraction info
- searchKnowledgeBase: Query the RAGFlow travel knowledge base for professional destination content
- geocode: Convert a city/address name to coordinates and adcode (use this to find adcode for weather)
- getWeather: Get real-time weather or forecast for a city (requires adcode from geocode or the table below)
- scrapeWeb: Scrape a specific URL for detailed content
- generatePDF: Generate a downloadable PDF travel guide
- fileOperation: Read/write files on disk

WORKFLOW:
1. geocode("destination city") to get the adcode for weather lookup
2. searchWeb("destination travel guide tips 2025")
3. searchKnowledgeBase("destination attractions food")
4. getWeather("city adcode") — use geocode first if you don't know the adcode
5. searchWeb again for specific topics (hotels, transportation, budget)
6. scrapeWeb any high-quality URLs found
7. generatePDF(fileName, full travel guide content) — ALWAYS do this before finishing
8. doTerminate

OUTPUT REQUIREMENTS:
Your final answer MUST be a COMPREHENSIVE travel guide (at least 700 Chinese characters) covering:
- Weather and best travel season
- Must-see attractions (at least 5, with ticket prices, opening hours, highlights)
- Food recommendations (at least 5 local specialties with suggested restaurants)
- Transportation tips (how to get there and get around)
- Suggested itinerary (day by day)
- Budget estimate
- Practical tips (local customs, what to pack, etc.)
Use markdown formatting with tables, headers, and emojis to make it readable.

After your text answer, call generatePDF with the FULL guide content, then call doTerminate.
Always respond in Chinese (简体中文)."""

TRAVEL_MANUS_NEXT_STEP_PROMPT = """\
CRITICAL: You MUST call multiple tools before answering. A short answer is NOT acceptable.

Required steps for ANY travel question:
- Step 1: geocode the destination city name to get its adcode (DO NOT use the MCP tools for geocoding!)
- Step 2: searchWeb with a detailed search query about the destination
- Step 3: searchKnowledgeBase with the same query
- Step 4: getWeather for the destination city using the adcode from Step 1 (common adcodes: 北京=110000, 上海=310000, 广州=440100, 深圳=440300, 成都=510100, 杭州=330100, 南京=320100, 武汉=420100, 西安=610100, 重庆=500000, 厦门=350200, 青岛=370200, 三亚=460200, 大理=532900, 丽江=530700)
- Step 5: searchWeb again for specific topics (food, hotels, budget)
- Step 6: scrapeWeb for detailed content from key URLs
- Step 7: generatePDF with a comprehensive travel guide document
- Step 8: doTerminate

You MUST generate a PDF before terminating.
Do NOT give a short answer — the guide must be at least 1500 characters long.
After each tool execution, analyze the results and search for more if needed."""

TRAVEL_APP_SYSTEM_PROMPT = "你是 AI 旅游搭子，一位专业的智能旅行规划师。" \
    "开场向用户热情打招呼，表明自己是「AI旅游搭子」，可以帮用户规划旅行。" \
    "主动询问用户的旅行目的地、出行时间、预算范围、兴趣爱好（美食/文化/自然/购物/亲子等）出行人数及同伴关系。" \
    "根据用户提供的信息，结合网络搜索和专业知识库，" \
    "为用户推荐热门景点、特色美食、住宿建议、交通方式，并生成详细的行程安排。" \
    "能够调用工具生成精美的 PDF 旅游攻略供用户下载保存。"
```

---

### Task 3: Simple Tools (terminate, file_operation, resource_download)

**Files:**
- Create: `src/tools/__init__.py`
- Create: `src/tools/terminate.py`
- Create: `src/tools/file_operation.py`
- Create: `src/tools/resource_download.py`

**Interfaces:**
- Produces: `do_terminate()`, `read_file()`, `write_file()`, `download_resource()` — all LangChain @tool functions

- [ ] **Step 1: Create src/tools/terminate.py**

```python
from langchain.tools import tool

@tool(description="""Terminate the interaction when the request is met OR if the assistant cannot proceed further with the task.
When you have finished all the tasks, call this tool to end the work.""")
def do_terminate() -> str:
    return "任务结束"
```

- [ ] **Step 2: Create src/tools/file_operation.py**

```python
from pathlib import Path
from langchain.tools import tool

FILE_DIR = Path.cwd() / "tmp" / "file"

@tool(description="Read content from a file")
def read_file(file_name: str) -> str:
    file_path = FILE_DIR / file_name
    try:
        return file_path.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading file: {e}"

@tool(description="Write content to a file")
def write_file(file_name: str, content: str) -> str:
    file_path = FILE_DIR / file_name
    try:
        FILE_DIR.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return f"File written successfully to: {file_path}"
    except Exception as e:
        return f"Error writing to file: {e}"
```

- [ ] **Step 3: Create src/tools/resource_download.py**

```python
from pathlib import Path
import httpx
from langchain.tools import tool

DOWNLOAD_DIR = Path.cwd() / "tmp" / "download"

@tool(description="Download a resource from a given URL")
def download_resource(url: str, file_name: str) -> str:
    file_path = DOWNLOAD_DIR / file_name
    try:
        DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
        with httpx.Client(timeout=30) as client:
            response = client.get(url)
            response.raise_for_status()
        file_path.write_bytes(response.content)
        return f"Resource downloaded successfully to: {file_path}"
    except Exception as e:
        return f"Error downloading resource: {e}"
```

- [ ] **Step 4: Create minimal src/tools/__init__.py** (will be expanded in Task 8)

```python
from src.tools.terminate import do_terminate
from src.tools.file_operation import read_file, write_file
from src.tools.resource_download import download_resource

LOCAL_TOOLS = [do_terminate, read_file, write_file, download_resource]
```

---

### Task 4: Search Tools (tavily, ragflow)

**Files:**
- Create: `src/tools/tavily_search.py`
- Create: `src/tools/ragflow_search.py`

**Interfaces:**
- Produces: `search_web(query)` → str, `search_knowledge_base(query)` → str
- Consumes: `settings` from `src.config`

- [ ] **Step 1: Create src/tools/tavily_search.py**

```python
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
```

- [ ] **Step 2: Create src/tools/ragflow_search.py**

```python
import httpx
from langchain.tools import tool
from src.config import settings
from src.rag.query_expander import expand_query, deduplicate_results
import logging

logger = logging.getLogger(__name__)

@tool(description="Search the RAGFlow knowledge base for professional travel information, guides, and destination details")
def search_knowledge_base(query: str) -> str:
    expanded = expand_query(query)
    if len(expanded) <= 1:
        return _search_single(expanded[0])

    logger.info("RAGFlow 多查询扩展检索：原始查询='%s'，扩展为 %d 个子查询", query, len(expanded))
    all_lines = []
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
            all_lines.append(f"=== {sub_query} ===\n{deduped_str}\n")
            found_any = True

    if found_any:
        return "\n".join(all_lines).strip()[:3000]
    logger.warning("所有扩展查询均未返回结果，兜底使用原始查询")
    return _search_single(query) or "RAGFlow 知识库中未找到相关信息。"

def _search_single(query: str) -> str:
    try:
        url = f"{settings.ragflow_base_url}/api/v1/retrieval"
        response = httpx.post(
            url,
            json={"question": query, "kb_ids": [settings.ragflow_kb_id], "page": 1, "size": 10},
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
        return "\n".join(lines) if lines else "RAGFlow 知识库中未找到相关信息。"
    except Exception as e:
        logger.error("RAGFlow 检索异常, query='%s': %s", query, e)
        return f"RAGFlow 知识库检索出错：{e}"
```

---

### Task 5: Amap Tools (geocode, weather)

**Files:**
- Create: `src/tools/amap_geocode.py`
- Create: `src/tools/amap_weather.py`

**Interfaces:**
- Produces: `geocode(address, city=None)` → str, `get_weather(city_code, type="base")` → str

- [ ] **Step 1: Create src/tools/amap_geocode.py**

```python
import httpx
from langchain.tools import tool
from src.config import settings

@tool(description="将地址（城市名/地标/景点名）转换为经纬度和adcode城市编码。用于获取城市的adcode以查询天气")
def geocode(address: str, city: str = None) -> str:
    try:
        params = {"key": settings.amap_api_key, "address": address, "output": "JSON"}
        if city:
            params["city"] = city
        response = httpx.get("https://restapi.amap.com/v3/geocode/geo", params=params, timeout=10)
        response.raise_for_status()
        return response.text[:3000]
    except Exception as e:
        return f"地理编码失败：{e}"

@tool(description="查询指定城市的实时天气或天气预报。cityCode 为高德城市编码（adcode），如北京=110000。type=base 获取实时天气，type=all 获取未来4天预报")
def get_weather(city_code: str, type: str = "base") -> str:
    try:
        params = {"key": settings.amap_api_key, "city": city_code, "extensions": type, "output": "JSON"}
        response = httpx.get("https://restapi.amap.com/v3/weather/weatherInfo", params=params, timeout=10)
        response.raise_for_status()
        return response.text[:3000]
    except Exception as e:
        return f"天气查询失败：{e}"
```

Wait — I should put both in one file or separate? The design spec says separate files. Let me keep them separate.

Actually, let me reconsider. The user wants to minimize overhead. Two tiny files that both call the same API... Let me follow the spec exactly. Separate files.

- [ ] **Step 2: Create src/tools/amap_weather.py**

```python
import httpx
from langchain.tools import tool
from src.config import settings

@tool(description="查询指定城市的实时天气或天气预报。cityCode 为高德城市编码（adcode），如北京=110000。type=base 获取实时天气，type=all 获取未来4天预报")
def get_weather(city_code: str, type: str = "base") -> str:
    try:
        params = {"key": settings.amap_api_key, "city": city_code, "extensions": type, "output": "JSON"}
        response = httpx.get("https://restapi.amap.com/v3/weather/weatherInfo", params=params, timeout=10)
        response.raise_for_status()
        return response.text[:3000]
    except Exception as e:
        return f"天气查询失败：{e}"
```

---

### Task 6: Web Scraping + PDF Generation

**Files:**
- Create: `src/tools/web_scraping.py`
- Create: `src/tools/pdf_generation.py`

**Interfaces:**
- Produces: `scrape_web(url)` → str, `generate_pdf(file_name, content)` → str

- [ ] **Step 1: Create src/tools/web_scraping.py**

```python
from bs4 import BeautifulSoup
import httpx
from langchain.tools import tool

@tool(description="Scrape the content of a web page")
def scrape_web(url: str) -> str:
    try:
        response = httpx.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        return text[:3000]
    except Exception as e:
        return f"Error scraping web page: {e}"
```

- [ ] **Step 2: Create src/tools/pdf_generation.py**

```python
from pathlib import Path
from langchain.tools import tool
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from src.config import settings
import logging

logger = logging.getLogger(__name__)

PDF_DIR = Path.cwd() / "tmp" / "pdf"

# Register Chinese font — STSong or a bundled TTF
_FONT_REGISTERED = False

def _ensure_font():
    global _FONT_REGISTERED
    if _FONT_REGISTERED:
        return
    font_paths = [
        "C:/Windows/Fonts/STSONG.TTF",
        "C:/Windows/Fonts/simsun.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for fp in font_paths:
        if Path(fp).exists():
            pdfmetrics.registerFont(TTFont("ChineseFont", fp))
            _FONT_REGISTERED = True
            return
    logger.warning("No Chinese font found, PDF may not render Chinese correctly")


def _create_local_pdf(file_path: Path, content: str):
    _ensure_font()
    doc = SimpleDocTemplate(str(file_path), pagesize=A4)
    styles = getSampleStyleSheet()
    style = styles["Normal"]
    style.fontName = "ChineseFont" if _FONT_REGISTERED else "Helvetica"
    style.fontSize = 10
    style.leading = 14
    story = [Paragraph(content.replace("\n", "<br/>"), style)]
    doc.build(story)


def _upload_to_cos(file_path: Path, file_name: str) -> str:
    if not all([settings.cos_secret_id, settings.cos_secret_key, settings.cos_bucket_name]):
        raise RuntimeError("COS 配置不完整")
    from qcloud_cos import CosConfig, CosS3Client
    config = CosConfig(Region=settings.cos_region, SecretId=settings.cos_secret_id,
                       SecretKey=settings.cos_secret_key)
    client = CosS3Client(config)
    object_key = f"pdf/{file_name}"
    client.put_object_from_local_file(
        Bucket=settings.cos_bucket_name,
        LocalFilePath=str(file_path),
        Key=object_key,
    )
    return f"https://{settings.cos_bucket_name}.cos.{settings.cos_region}.myqcloud.com/{object_key}"


@tool(description="Generate a PDF file with given content")
def generate_pdf(file_name: str, content: str) -> str:
    file_path = PDF_DIR / file_name
    try:
        PDF_DIR.mkdir(parents=True, exist_ok=True)
        _create_local_pdf(file_path, content)
        if not file_path.exists() or file_path.stat().st_size == 0:
            return "Error: Failed to create local PDF file"
        cos_url = _upload_to_cos(file_path, file_name)
        return f"PDF 文件已成功生成并上传到云存储。文件名： {file_name}，用户可以使用该文件名访问或下载此文档。"
    except Exception as e:
        logger.error("生成 PDF 时发生错误：%s", e)
        return f"Error generating PDF: {e}"
```

---

### Task 7: RAG Query Expander + MCP Client

**Files:**
- Create: `src/rag/__init__.py`
- Create: `src/rag/query_expander.py`
- Create: `src/mcp/__init__.py`
- Create: `src/mcp/client.py`

**Interfaces:**
- Produces: `expand_query(query)` → list[str], `get_mcp_client()` → MultiServerMCPClient
- Consumes: `settings`

- [ ] **Step 1: Create src/rag/query_expander.py**

```python
"""
Multi-query expansion for RAGFlow retrieval.
Uses simple keyword-based expansion to break a travel query into sub-queries.
"""

def expand_query(query: str) -> list[str]:
    aspects = ["景点", "美食", "交通", "住宿", "天气", "攻略", "门票", "文化"]
    expanded = [query]
    for aspect in aspects:
        expanded.append(f"{query} {aspect}")
    return expanded[:5]
```

- [ ] **Step 2: Create src/mcp/client.py**

```python
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from src.config import settings

_mcp_client = None

def get_mcp_config() -> dict:
    return {
        "amap-maps": {
            "command": "npx",
            "args": ["-y", "@amap/amap-maps-mcp-server"],
            "env": {"AMAP_MAPS_API_KEY": settings.amap_api_key},
            "transport": "stdio",
        },
        "yu-image-search": {
            "command": "python",
            "args": ["-m", "src.image_search_mcp.main"],
            "env": {"PEXELS_API_KEY": settings.pexels_api_key},
            "transport": "stdio",
        },
    }

def create_mcp_client() -> MultiServerMCPClient:
    return MultiServerMCPClient(get_mcp_config())
```

---

### Task 8: Tool Registration (wiring all tools together)

**Files:**
- Modify: `src/tools/__init__.py`

**Interfaces:**
- Produces: `get_local_tools()` → list, `get_all_tools(mcp_client)` → list

- [ ] **Step 1: Rewrite src/tools/__init__.py**

```python
from src.tools.terminate import do_terminate
from src.tools.file_operation import read_file, write_file
from src.tools.resource_download import download_resource
from src.tools.tavily_search import search_web
from src.tools.ragflow_search import search_knowledge_base
from src.tools.amap_geocode import geocode
from src.tools.amap_weather import get_weather
from src.tools.web_scraping import scrape_web
from src.tools.pdf_generation import generate_pdf

def get_local_tools():
    return [
        do_terminate,
        read_file,
        write_file,
        download_resource,
        search_web,
        search_knowledge_base,
        geocode,
        get_weather,
        scrape_web,
        generate_pdf,
    ]

def get_all_tools(mcp_client=None):
    tools = list(get_local_tools())
    if mcp_client is not None:
        try:
            tools.extend(list(mcp_client.get_tools()))
        except Exception:
            pass
    return tools
```

---

### Task 9: Chat Memory

**Files:**
- Create: `src/memory/__init__.py`
- Create: `src/memory/file_memory.py`

**Interfaces:**
- Produces: `FileBasedChatMemory` class with `get()`, `add()`, `clear()` methods
- NOTE: We use LangChain's `InMemoryChatMessageHistory` pattern but persist with pickle

- [ ] **Step 1: Create src/memory/file_memory.py**

```python
import pickle
from pathlib import Path
from langchain.memory import ConversationBufferMemory

class FileBasedChatMemory(ConversationBufferMemory):
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        super().__init__(memory_key="history", return_messages=True)

    def _file_path(self, conversation_id: str) -> Path:
        safe_id = conversation_id.replace("/", "_").replace("\\", "_")
        return self.base_dir / f"{safe_id}.pickle"

    def save(self, conversation_id: str):
        file_path = self._file_path(conversation_id)
        with open(file_path, "wb") as f:
            pickle.dump(self.chat_memory.messages, f)

    def load(self, conversation_id: str):
        file_path = self._file_path(conversation_id)
        if file_path.exists():
            with open(file_path, "rb") as f:
                messages = pickle.load(f)
            self.chat_memory.messages = messages
            return messages
        return []

    def clear_conversation(self, conversation_id: str):
        file_path = self._file_path(conversation_id)
        if file_path.exists():
            file_path.unlink()
        self.chat_memory.clear()
```

Wait, this is cleaner:

```python
import pickle
from pathlib import Path
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict

class FileBasedChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, base_dir: str, conversation_id: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        safe_id = conversation_id.replace("/", "_").replace("\\", "_")
        self.file_path = self.base_dir / f"{safe_id}.pickle"
        self._messages: list[BaseMessage] = []

    @property
    def messages(self) -> list[BaseMessage]:
        self._ensure_loaded()
        return self._messages

    def add_messages(self, messages: list[BaseMessage]) -> None:
        self._ensure_loaded()
        self._messages.extend(messages)
        self._save()

    def clear(self) -> None:
        self._messages = []
        if self.file_path.exists():
            self.file_path.unlink()

    def _ensure_loaded(self):
        if not self._messages and self.file_path.exists():
            with open(self.file_path, "rb") as f:
                self._messages = pickle.load(f)

    def _save(self):
        with open(self.file_path, "wb") as f:
            pickle.dump(self._messages, f)
```

---

### Task 10: Agent Graph (LangGraph StateGraph)

**Files:**
- Create: `src/agent/graph.py`

**Interfaces:**
- Produces: `build_travel_manus_graph(chat_model, tools)` → compiled StateGraph
- Consumes: `AgentState` from `state.py`, prompts from `prompts.py`

- [ ] **Step 1: Create src/agent/graph.py**

```python
import logging
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from src.agent.state import AgentState
from src.agent.prompts import TRAVEL_MANUS_SYSTEM_PROMPT, TRAVEL_MANUS_NEXT_STEP_PROMPT

logger = logging.getLogger(__name__)

def _model_call(state: AgentState, llm_with_tools) -> dict:
    """Call LLM with bound tools. Returns updated messages."""
    messages = state["messages"]
    # Inject next_step_prompt as a user message if not first call
    if state["current_step"] > 0 and state.get("next_step_prompt"):
        messages = list(messages) + [HumanMessage(content=state["next_step_prompt"])]
    response = llm_with_tools.invoke(messages)
    return {
        "messages": [response],
        "current_step": state["current_step"] + 1,
    }


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


def _should_loop(state: AgentState) -> str:
    if state.get("agent_state") in ("FINISHED", "ERROR"):
        return "finalize"
    if state.get("current_step", 0) >= state.get("max_steps", 20):
        return "finalize"
    # Check if terminate was called
    messages = state["messages"]
    for msg in messages:
        if hasattr(msg, "tool_calls"):
            continue
        if hasattr(msg, "content") and "任务结束" in str(getattr(msg, "content", "")):
            return "finalize"
    return "thinker"


def _check_state(state: AgentState) -> dict:
    if state.get("agent_state") != "RUNNING":
        return {"agent_state": "RUNNING"}
    return {}


def _finalize(state: AgentState) -> dict:
    return {"agent_state": "FINISHED"}


def build_travel_manus_graph(chat_model, tools):
    llm_with_tools = chat_model.bind_tools(tools)

    def thinker(state: AgentState) -> dict:
        return _model_call(state, llm_with_tools)

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


def build_initial_state(user_message: str) -> AgentState:
    return AgentState(
        messages=[
            SystemMessage(content=TRAVEL_MANUS_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ],
        agent_state="RUNNING",
        current_step=0,
        max_steps=20,
        system_prompt=TRAVEL_MANUS_SYSTEM_PROMPT,
        next_step_prompt=TRAVEL_MANUS_NEXT_STEP_PROMPT,
    )
```

---

### Task 11: API Router

**Files:**
- Create: `src/api/__init__.py`
- Create: `src/api/schemas.py`
- Create: `src/api/router.py`

**Interfaces:**
- Produces: FastAPI APIRouter with 4 endpoints
- Consumes: chat_model, tools, chat_memory, agent_graph from app state

- [ ] **Step 1: Create src/api/schemas.py**

```python
from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str
    chat_id: str = "default"
```

- [ ] **Step 2: Create src/api/router.py**

```python
import json
import logging
from fastapi import APIRouter, Request, Query
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, SystemMessage
from src.agent.graph import build_travel_manus_graph, build_initial_state
from src.agent.prompts import TRAVEL_APP_SYSTEM_PROMPT

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/ai")


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
    memory = request.app.state.chat_memory
    messages = memory.load(chat_id)
    messages.append(HumanMessage(content=message))

    response = chat_model.invoke(messages)
    messages.append(response)
    memory.add_messages([response])
    memory.save(chat_id)
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
        full_content = ""
        try:
            async for chunk in chat_model.astream(messages):
                if chunk.content:
                    full_content += chunk.content
                    yield f"data: {json.dumps(chunk.content, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error("SSE error: %s", e)
            yield f"data: [ERROR] {e}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/travel/agent/chat")
async def travel_agent_chat(
    request: Request,
    message: str = Query(...),
):
    chat_model = request.app.state.chat_model
    tools = request.app.state.tools
    agent_graph = build_travel_manus_graph(chat_model, tools)
    initial_state = build_initial_state(message)

    async def event_stream():
        try:
            async for event in agent_graph.astream_events(initial_state, version="v2"):
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
```

---

### Task 12: Main App Entry Point

**Files:**
- Create: `src/main.py`

**Interfaces:**
- Produces: FastAPI `app` instance with lifespan, mount all routers

- [ ] **Step 1: Create src/main.py**

```python
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from langchain_openai import ChatOpenAI
from src.config import settings
from src.tools import get_all_tools
from src.api.router import router as api_router
from src.web.router import router as web_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting AI Tourism app...")
    app.state.chat_model = ChatOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.chat_model,
        temperature=settings.temperature,
        streaming=True,
    )
    app.state.tools = get_all_tools()
    logger.info("App started with %d tools", len(app.state.tools))
    yield
    logger.info("Shutting down...")


app = FastAPI(title="AI旅游搭子", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
app.include_router(web_router)

# Mount static files (not in lifespan; StaticFiles is a router)
import os
static_dir = os.path.join(os.path.dirname(__file__), "web", "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
```

---

### Task 13: Image Search MCP Server

**Files:**
- Create: `src/image_search_mcp/__init__.py`
- Create: `src/image_search_mcp/main.py`

**Interfaces:**
- Produces: stdio MCP server with `search_image` tool

- [ ] **Step 1: Create src/image_search_mcp/main.py**

```python
import os
import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server

server = Server("yu-image-search")
API_URL = "https://api.pexels.com/v1/search"

@server.tool()
async def search_image(query: str) -> str:
    api_key = os.environ.get("PEXELS_API_KEY", "")
    try:
        response = httpx.get(
            API_URL,
            params={"query": query, "per_page": 5},
            headers={"Authorization": api_key},
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
        photos = data.get("photos", [])
        urls = [p.get("src", {}).get("medium", "") for p in photos]
        urls = [u for u in urls if u]
        return ",".join(urls) if urls else "未找到相关图片"
    except Exception as e:
        return f"Error search image: {e}"

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

---

### Task 14: Frontend — Templates + CSS

**Files:**
- Create: `src/web/__init__.py`
- Create: `src/web/router.py`
- Create: `src/web/templates/base.html`
- Create: `src/web/templates/index.html`
- Create: `src/web/templates/travel.html`
- Create: `src/web/static/style.css`
- Create: `src/web/static/app.js`

**Interfaces:**
- Produces: Jinja2 templates serving `/` and `/travel` pages

- [ ] **Step 1: Create src/web/router.py**

```python
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

router = APIRouter()
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@router.get("/travel", response_class=HTMLResponse)
async def travel(request: Request):
    return templates.TemplateResponse("travel.html", {"request": request})
```

- [ ] **Step 2: Create src/web/templates/base.html**

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}AI旅游搭子{% endblock %}</title>
    <link rel="stylesheet" href="/static/style.css">
    <script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.14.1/dist/cdn.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    {% block head %}{% endblock %}
</head>
<body>
    {% block content %}{% endblock %}
</body>
</html>
```

- [ ] **Step 3: Create src/web/templates/index.html**

```html
{% extends "base.html" %}
{% block title %}AI旅游搭子 — 首页{% endblock %}
{% block content %}
<div class="landing">
    <h1>🧳 AI旅游搭子</h1>
    <p class="subtitle">智能旅行规划师 — 联网搜索、深度攻略、PDF 下载</p>
    <div class="cards">
        <a href="/travel" class="card">
            <h2>🤖 AI旅游超级智能体</h2>
            <p>多工具自主规划，生成完整旅行攻略</p>
        </a>
    </div>
</div>
{% endblock %}
```

- [ ] **Step 4: Create src/web/templates/travel.html** — the main chat page with SSE + Alpine.js

```html
{% extends "base.html" %}
{% block title %}AI旅游超级智能体{% endblock %}
{% block content %}
<div x-data="chatApp()" class="travel-page">
    <header class="header">
        <a href="/" class="back-btn">&larr; 返回</a>
        <h1>AI旅游超级智能体</h1>
        <span></span>
    </header>
    <div class="chat-container" x-ref="container">
        <div class="messages" x-ref="messages">
            <template x-for="msg in messages" :key="msg.time">
                <div :class="msg.isUser ? 'msg user' : 'msg ai'">
                    <div class="bubble" x-html="msg.html || msg.content"></div>
                    <div class="time" x-text="msg.timeStr"></div>
                </div>
            </template>
            <div x-show="loading" class="msg ai">
                <div class="bubble typing">▋</div>
            </div>
        </div>
        <form class="input-area" @submit.prevent="send">
            <textarea x-model="input" @keydown.enter.prevent="send" placeholder="输入你想去的旅行目的地……" :disabled="loading"></textarea>
            <button type="submit" :disabled="loading || !input.trim()">发送</button>
        </form>
    </div>
</div>
<script>
function chatApp() {
    return {
        messages: [],
        input: '',
        loading: false,
        eventSource: null,
        buffer: '',
        lastBubbleTime: 0,

        init() {
            this.addMsg('你好！我是 AI旅游超级智能体。我可以联网搜索最新旅游攻略、查询专业知识库、规划行程、生成PDF攻略。告诉我你想去哪里，我来帮你搞定一切！', false);
        },

        addMsg(content, isUser) {
            let html = content;
            try { html = marked.parse(content); } catch(e) {}
            const now = Date.now();
            this.messages.push({
                content: content,
                html: html,
                isUser: isUser,
                time: now,
                timeStr: new Date(now).toLocaleTimeString('zh-CN', {hour:'2-digit',minute:'2-digit'})
            });
            this.$nextTick(() => {
                const el = this.$refs.messages;
                el.scrollTop = el.scrollHeight;
            });
        },

        send() {
            const msg = this.input.trim();
            if (!msg || this.loading) return;
            this.addMsg(msg, true);
            this.input = '';
            this.loading = true;
            this.buffer = '';
            this.lastBubbleTime = Date.now();

            if (this.eventSource) this.eventSource.close();
            const url = `/api/ai/travel/agent/chat?message=${encodeURIComponent(msg)}`;
            this.eventSource = new EventSource(url);

            this.eventSource.onmessage = (e) => {
                const data = e.data;
                if (data === '[DONE]') {
                    if (this.buffer) this.flushBuffer();
                    this.loading = false;
                    this.eventSource.close();
                    return;
                }
                if (data.startsWith('[TOOL]')) {
                    this.buffer += data + '\n';
                    return;
                }
                if (data.startsWith('[ERROR]')) {
                    this.addMsg('❌ ' + data, false);
                    this.loading = false;
                    this.eventSource.close();
                    return;
                }
                try {
                    const text = JSON.parse(data);
                    this.buffer += text;
                    const lastChar = text.slice(-1);
                    if ('。！？…'.includes(lastChar) || this.buffer.length > 500) {
                        this.flushBuffer();
                    }
                } catch(e) {
                    this.buffer += data;
                }
            };

            this.eventSource.onerror = () => {
                if (this.buffer) this.flushBuffer();
                this.loading = false;
                this.eventSource.close();
            };
        },

        flushBuffer() {
            if (!this.buffer.trim()) return;
            const now = Date.now();
            const elapsed = now - this.lastBubbleTime;
            if (elapsed < 800) {
                setTimeout(() => this.flushBuffer(), 800 - elapsed);
                return;
            }
            const content = this.buffer.replace(/^Step\s*\d+\s*:\s*/gm, '').trim();
            this.addMsg(content, false);
            this.buffer = '';
            this.lastBubbleTime = Date.now();
        }
    }
}
</script>
{% endblock %}
```

- [ ] **Step 5: Create src/web/static/style.css** — full CSS replicating the original Vue warm-tone design

```css
:root {
    --bg-1: #1a1a2e;
    --bg-2: #16213e;
    --coral: #E8734A;
    --coral-glow: rgba(232, 115, 74, 0.3);
    --text-main: #e0e0e0;
    --text-soft: #aaa;
    --hairline: rgba(255,255,255,0.08);
}

* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans SC", sans-serif;
    background: var(--bg-1);
    color: var(--text-main);
    min-height: 100vh;
}

/* Landing */
.landing { text-align: center; padding: 80px 20px; }
.landing h1 { font-size: 2.5rem; margin-bottom: 12px; }
.subtitle { color: var(--text-soft); font-size: 1.1rem; margin-bottom: 48px; }
.cards { display: flex; justify-content: center; gap: 24px; flex-wrap: wrap; }
.card {
    display: block; background: var(--bg-2); border: 1px solid var(--hairline);
    border-radius: 16px; padding: 32px 28px; text-decoration: none; color: var(--text-main);
    transition: transform 0.2s, border-color 0.2s; max-width: 360px;
}
.card:hover { transform: translateY(-2px); border-color: var(--coral); }
.card h2 { font-size: 1.3rem; margin-bottom: 8px; }
.card p { color: var(--text-soft); font-size: 0.95rem; }

/* Chat Page */
.travel-page { display: flex; flex-direction: column; height: 100vh; }
.header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 24px; background: var(--bg-2); border-bottom: 1px solid var(--hairline);
}
.header h1 { font-size: 1.15rem; font-weight: 600; }
.back-btn { color: var(--text-soft); text-decoration: none; font-size: 0.95rem; }
.back-btn:hover { color: var(--coral); }

.chat-container {
    flex: 1; display: flex; flex-direction: column; overflow: hidden;
    max-width: 900px; margin: 0 auto; width: 100%;
}

.messages {
    flex: 1; overflow-y: auto; padding: 20px 18px 80px;
    display: flex; flex-direction: column; gap: 14px;
}
.msg { display: flex; flex-direction: column; max-width: 85%; animation: fadeIn 0.25s ease; }
.msg.user { align-self: flex-end; }
.msg.ai { align-self: flex-start; }
.bubble {
    padding: 12px 16px; border-radius: 16px; line-height: 1.6; font-size: 0.95rem;
    word-wrap: break-word;
}
.msg.ai .bubble { background: var(--bg-2); border: 1px solid var(--hairline); border-bottom-left-radius: 6px; }
.msg.user .bubble { background: rgba(232,115,74,0.18); border: 1px solid rgba(232,115,74,0.2); border-bottom-right-radius: 6px; }
.msg.ai .bubble h1, .msg.ai .bubble h2, .msg.ai .bubble h3 { margin: 8px 0 4px; }
.msg.ai .bubble table { width: 100%; border-collapse: collapse; margin: 8px 0; font-size: 0.85rem; }
.msg.ai .bubble th, .msg.ai .bubble td { border: 1px solid var(--hairline); padding: 6px 10px; text-align: left; }
.msg.ai .bubble th { background: rgba(232,115,74,0.12); }
.time { font-size: 11px; opacity: 0.5; margin-top: 6px; text-align: right; }
.msg.ai .time { text-align: left; }

.typing { animation: blink 0.7s infinite; color: var(--coral); }

.input-area {
    display: flex; padding: 14px 16px; background: var(--bg-2);
    border-top: 1px solid var(--hairline); gap: 10px;
}
.input-area textarea {
    flex: 1; border: 1px solid var(--hairline); border-radius: 14px;
    padding: 10px 16px; font-size: 0.95rem; resize: none; outline: none;
    background: var(--bg-1); color: var(--text-main); font-family: inherit;
    min-height: 20px; max-height: 120px;
}
.input-area textarea:focus { border-color: var(--coral); }
.input-area button {
    padding: 0 22px; background: var(--coral); color: #fff; border: none;
    border-radius: 14px; font-size: 0.95rem; font-weight: 500; cursor: pointer;
    transition: filter 0.2s;
}
.input-area button:hover:not(:disabled) { filter: brightness(1.1); }
.input-area button:disabled { opacity: 0.4; cursor: not-allowed; }

@keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
@keyframes blink { 0% { opacity: 0; } 50% { opacity: 1; } 100% { opacity: 0; } }

@media (max-width: 768px) {
    .msg { max-width: 92%; }
    .header { padding: 12px 16px; }
    .header h1 { font-size: 1rem; }
}
```

- [ ] **Step 6: Create src/web/static/app.js** — empty placeholder for shared JS

```js
// Shared JS utilities (extensible for future pages)
console.log('AI旅游搭子 loaded');
```

---

### Task 15: Dockerfile

**Files:**
- Create: `Dockerfile`

- [ ] **Step 1: Create Dockerfile**

```dockerfile
FROM python:3.12-slim
WORKDIR /app

# Install uv
RUN pip install uv

# Install deps
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy source
COPY src/ src/

EXPOSE 8123
CMD ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8123"]
```

---

### Task 16: Integration Test & Verify

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_tools.py`
- Create: `tests/test_agent.py`

- [ ] **Step 1: Create tests/test_tools.py**

```python
from src.tools.terminate import do_terminate
from src.tools.file_operation import read_file, write_file

def test_terminate():
    assert do_terminate() == "任务结束"

def test_write_read_file():
    result = write_file("test.txt", "hello")
    assert "successfully" in result
    content = read_file("test.txt")
    assert content == "hello"
```

- [ ] **Step 2: Create tests/test_agent.py**

```python
from src.agent.state import AgentState

def test_agent_state_defaults():
    state = AgentState(
        messages=[],
        agent_state="IDLE",
        current_step=0,
        max_steps=20,
        system_prompt="test",
        next_step_prompt="test",
    )
    assert state["agent_state"] == "IDLE"
    assert state["max_steps"] == 20
```

- [ ] **Step 3: Run tests**

```bash
cd E:/000AI_tourism_Python
uv run pytest tests/ -v
```

---

### Task 17: Verify Startup

- [ ] **Step 1: Start the server**

```bash
cd E:/000AI_tourism_Python
uv run uvicorn src.main:app --host 0.0.0.0 --port 8123
```

- [ ] **Step 2: Test health endpoint**

Open browser or curl: `http://localhost:8123/api/ai/health`
Expected: `{"status": "ok"}`

- [ ] **Step 3: Test web pages**

Open: `http://localhost:8123/` → landing page
Open: `http://localhost:8123/travel` → chat page
