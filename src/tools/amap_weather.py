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
