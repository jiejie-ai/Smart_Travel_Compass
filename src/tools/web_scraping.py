import httpx
from bs4 import BeautifulSoup
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
