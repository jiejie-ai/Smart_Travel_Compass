from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from jinja2 import Environment, FileSystemLoader

router = APIRouter()

templates_dir = Path(__file__).parent / "templates"
env = Environment(loader=FileSystemLoader(str(templates_dir)), auto_reload=False)


def _render(template_name: str, request: Request) -> HTMLResponse:
    template = env.get_template(template_name)
    content = template.render(request=request)
    return HTMLResponse(content=content)


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return _render("index.html", request)


@router.get("/travel", response_class=HTMLResponse)
async def travel(request: Request):
    return _render("travel.html", request)
