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
