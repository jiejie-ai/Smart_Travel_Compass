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
