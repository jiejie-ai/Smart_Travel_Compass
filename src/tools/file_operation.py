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
