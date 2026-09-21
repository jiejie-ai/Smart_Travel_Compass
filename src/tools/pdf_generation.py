import logging
from io import BytesIO
from pathlib import Path

import httpx
from langchain.tools import tool
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, SimpleDocTemplate, Paragraph

from src.config import settings

logger = logging.getLogger(__name__)

PDF_DIR = Path.cwd() / "tmp" / "pdf"

_FONT_REGISTERED = False


def _ensure_font():
    global _FONT_REGISTERED
    if _FONT_REGISTERED:
        return
    font_paths = [
        "C:/Windows/Fonts/STSONG.TTF",
        "C:/Windows/Fonts/simsun.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for fp in font_paths:
        if Path(fp).exists():
            pdfmetrics.registerFont(TTFont("ChineseFont", fp))
            _FONT_REGISTERED = True
            return
    logger.warning("No Chinese font found, PDF may not render Chinese correctly")


def _create_local_pdf(file_path: Path, content: str, image_urls: list[str] = None):
    _ensure_font()
    doc = SimpleDocTemplate(str(file_path), pagesize=A4)
    styles = getSampleStyleSheet()
    style = styles["Normal"]
    style.fontName = "ChineseFont" if _FONT_REGISTERED else "Helvetica"
    style.fontSize = 10
    style.leading = 14

    story = []
    # Text paragraphs
    for para_text in content.split("\n\n"):
        if para_text.strip():
            story.append(Paragraph(para_text.strip().replace("\n", "<br/>"), style))

    # Images
    if image_urls:
        story.append(Paragraph("<br/>", style))
        page_width = A4[0] - doc.leftMargin - doc.rightMargin
        for url in image_urls:
            url = url.strip()
            if not url:
                continue
            try:
                r = httpx.get(url, timeout=15)
                r.raise_for_status()
                img = Image(BytesIO(r.content))
                # Scale to fit page width
                aspect = img.imageHeight / img.imageWidth
                img.drawWidth = min(img.imageWidth, page_width)
                img.drawHeight = img.drawWidth * aspect
                story.append(img)
            except Exception as e:
                logger.warning("Failed to load image %s: %s", url, e)
                story.append(Paragraph(f"[图片加载失败]", style))

    doc.build(story)


def _upload_to_cos(file_path: Path, file_name: str) -> str:
    if not all([settings.cos_secret_id, settings.cos_secret_key, settings.cos_bucket_name]):
        raise RuntimeError("COS 配置不完整，请在 .env 中配置 COS 相关信息")
    from qcloud_cos import CosConfig, CosS3Client
    config = CosConfig(
        Region=settings.cos_region,
        SecretId=settings.cos_secret_id,
        SecretKey=settings.cos_secret_key,
    )
    client = CosS3Client(config)
    object_key = f"pdf/{file_name}"
    client.put_object_from_local_file(
        Bucket=settings.cos_bucket_name,
        LocalFilePath=str(file_path),
        Key=object_key,
    )
    return f"https://{settings.cos_bucket_name}.cos.{settings.cos_region}.myqcloud.com/{object_key}"


@tool(description="Generate a PDF file with given content. image_urls is a comma-separated list of image URLs to embed in the PDF.")
def generate_pdf(file_name: str, content: str, image_urls: str = "") -> str:
    file_path = PDF_DIR / file_name
    try:
        PDF_DIR.mkdir(parents=True, exist_ok=True)
        urls = [u.strip() for u in image_urls.split(",") if u.strip()] if image_urls else None
        _create_local_pdf(file_path, content, urls)
        if not file_path.exists() or file_path.stat().st_size == 0:
            return "Error: Failed to create local PDF file"
        _upload_to_cos(file_path, file_name)
        img_count = len(urls) if urls else 0
        return f"PDF 文件已成功生成并上传到云存储（包含 {img_count} 张图片）。文件名：{file_name}"
    except Exception as e:
        logger.error("生成 PDF 时发生错误：%s", e)
        return f"Error generating PDF: {e}"
