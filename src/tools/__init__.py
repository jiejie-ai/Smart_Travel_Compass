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


# 提示词里可以引用的工具名白名单，供 tests/test_prompts.py 校验一致性。
LOCAL_TOOL_NAMES = frozenset(tool.name for tool in get_local_tools())

# amap-maps MCP 的工具名由第三方 npm 包在运行期提供，无法静态枚举，故提示词不引用它们。
# 若 src/image_search_mcp/main.py 里的工具改名，这里必须同步。
EXTERNAL_TOOL_NAMES = frozenset({"search_image"})
