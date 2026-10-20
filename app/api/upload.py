"""Bounded upload reads.

`app.main.MAX_UPLOAD_BYTES` stays the public override point (tests monkeypatch it),
so resolve the limit lazily at request time rather than binding the import value.
"""

import sys

from fastapi import HTTPException, UploadFile

from app.config import MAX_UPLOAD_BYTES


def current_max_upload_bytes() -> int:
    main_module = sys.modules.get("app.main")
    return getattr(main_module, "MAX_UPLOAD_BYTES", MAX_UPLOAD_BYTES)


async def read_upload_bounded(
    upload: UploadFile, max_bytes: int = MAX_UPLOAD_BYTES
) -> bytes:
    """Stream-reads an UploadFile in 64KB chunks, aborting with HTTP 413 if oversized."""
    chunks = bytearray()
    chunk_size = 64 * 1024
    while True:
        piece = await upload.read(chunk_size)
        if not piece:
            break
        chunks.extend(piece)
        if len(chunks) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=(
                    "Uploaded file exceeds maximum allowed size of "
                    f"{max_bytes // (1024 * 1024)} MB."
                ),
            )
    return bytes(chunks)
