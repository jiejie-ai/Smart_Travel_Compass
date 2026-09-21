from langchain_mcp_adapters.client import MultiServerMCPClient
from src.config import settings


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
