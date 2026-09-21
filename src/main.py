import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from langchain_openai import ChatOpenAI

from src.config import settings
from src.tools import get_local_tools
from src.api.router import router as api_router
from src.web.router import router as web_router
from src.mcp.client import create_mcp_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
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

    # Local tools (tavily, ragflow, amap, pdf, etc.)
    tools = list(get_local_tools())

    # MCP tools (image search, amap-maps)
    try:
        mcp_client = create_mcp_client()
        mcp_tools = await mcp_client.get_tools()
        tools.extend(mcp_tools)
        logger.info("Loaded %d MCP tools: %s", len(mcp_tools),
                     [t.name for t in mcp_tools])
    except Exception as e:
        logger.warning("Failed to load MCP tools, using local tools only: %s", e)

    app.state.tools = tools
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

static_dir = Path(__file__).parent / "web" / "static"
if static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
