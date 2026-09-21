import os

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("yu-image-search")
API_URL = "https://api.pexels.com/v1/search"

# Chinese city → English name mapping for Pexels search accuracy
_CITY_EN_MAP = {
    "北京": "Beijing China",
    "上海": "Shanghai China",
    "广州": "Guangzhou China",
    "深圳": "Shenzhen China",
    "成都": "Chengdu China",
    "杭州": "Hangzhou China",
    "武汉": "Wuhan China",
    "西安": "Xi'an China",
    "南京": "Nanjing China",
    "重庆": "Chongqing China",
    "长沙": "Changsha China",
    "苏州": "Suzhou China",
    "天津": "Tianjin China",
    "厦门": "Xiamen China",
    "青岛": "Qingdao China",
    "大连": "Dalian China",
    "昆明": "Kunming China",
    "哈尔滨": "Harbin China",
    "三亚": "Sanya China",
    "丽江": "Lijiang China",
    "大理": "Dali China",
    "拉萨": "Lhasa China",
    "桂林": "Guilin China",
    "黄山": "Huangshan China",
    "洛阳": "Luoyang China",
    "敦煌": "Dunhuang China",
}


@mcp.tool(description="Search for travel photos and destination images. query: search keywords in any language. query_en: English search term (recommended for Chinese destinations, e.g. 'Tianjin travel'). Returns image URLs.")
async def search_image(query: str, query_en: str = "") -> str:
    api_key = os.environ.get("PEXELS_API_KEY", "")
    # Priority: query_en > city mapping > query + "China" (if Chinese) > original query
    if query_en:
        search_query = query_en
    else:
        mapped = _CITY_EN_MAP.get(query)
        if mapped:
            search_query = mapped
        elif any("一" <= c <= "鿿" for c in query):
            search_query = f"{query} China"
        else:
            search_query = query
    try:
        response = httpx.get(
            API_URL,
            params={"query": search_query, "per_page": 5},
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


if __name__ == "__main__":
    mcp.run(transport="stdio")
