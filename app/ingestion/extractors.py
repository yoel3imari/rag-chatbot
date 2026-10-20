"""Text extraction from HTML and documents."""

import io
import re
from pathlib import Path
from typing import List

from bs4 import BeautifulSoup
from pypdf import PdfReader


def clean_html_to_text(html_content: str, url: str = "") -> str:
    """Strips boilerplate HTML to maximize token density."""
    soup = BeautifulSoup(html_content, "html.parser")
    for tag in soup(
        [
            "script",
            "style",
            "nav",
            "header",
            "footer",
            "aside",
            "noscript",
            "svg",
            "form",
            "iframe",
            "button",
        ]
    ):
        tag.decompose()

    for element in soup.find_all(
        attrs={
            "role": re.compile(
                r"^(navigation|banner|contentinfo|complementary|search)$", re.I
            )
        }
    ):
        element.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else url
    main_node = soup.find("main") or soup.find("article") or soup.body or soup

    lines: List[str] = []
    if title:
        lines.append(f"# {title}\n")

    for element in main_node.find_all(
        ["h1", "h2", "h3", "h4", "p", "li", "pre", "code", "blockquote", "td", "th"]
    ):
        text = re.sub(r"\s+", " ", element.get_text(" ", strip=True)).strip()
        if not text or len(text) < 3:
            continue
        if element.name in ("h1", "h2", "h3", "h4"):
            prefix = "#" * int(element.name[1])
            lines.append(f"\n{prefix} {text}\n")
        elif element.name == "li":
            lines.append(f"- {text}")
        else:
            lines.append(text)

    if len(lines) <= 2:
        raw_text = main_node.get_text("\n", strip=True)
        raw_text = re.sub(r"\n{3,}", "\n\n", raw_text)
        raw_text = re.sub(r"[ \t]+", " ", raw_text)
        return raw_text.strip()

    deduped_lines: List[str] = []
    for line in lines:
        if not deduped_lines or deduped_lines[-1] != line:
            deduped_lines.append(line)

    joined = "\n".join(deduped_lines)
    return re.sub(r"\n{3,}", "\n\n", joined).strip()


def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    """Extracts clean text from .pdf, .txt, or .md files."""
    extension = Path(filename).suffix.lower()
    if extension == ".pdf":
        reader = PdfReader(io.BytesIO(file_bytes))
        pages_text: List[str] = []
        for index, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = re.sub(r"[ \t]+", " ", text).strip()
            if text:
                pages_text.append(f"[Page {index + 1}]\n{text}")
        return "\n\n".join(pages_text)

    text = file_bytes.decode("utf-8", errors="replace")
    return re.sub(r"\r\n", "\n", text).strip()
