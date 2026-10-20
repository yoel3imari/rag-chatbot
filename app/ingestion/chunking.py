"""Semantic text chunking."""

import re
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(text: str, chunk_size: int = 750, chunk_overlap: int = 120) -> List[str]:
    """Splits text into overlapping semantic chunks using LangChain's splitter."""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        strip_whitespace=True,
    )
    return [chunk for chunk in splitter.split_text(text) if chunk.strip()]
