"""First-run corpus seeding."""

import asyncio

from app.embeddings import embedding_engine
from app.ingestion import chunk_text
from app.sample_data import SAMPLE_COMPANY_HANDBOOK
from app.store import store
from app.vector_store import vector_store


async def seed_default_handbook_if_empty() -> None:
    """Indexes the sample company handbook when the knowledge base is empty."""
    documents = store.list_documents()
    if documents:
        return

    document = store.create_document_record(
        name="acme-handbook-2026.md",
        source_type="file",
        scope="global",
        file_size=len(SAMPLE_COMPANY_HANDBOOK.encode("utf-8")),
    )
    document_id = document["id"]
    chunks = chunk_text(SAMPLE_COMPANY_HANDBOOK)
    vectors = await asyncio.to_thread(embedding_engine.embed_batch, chunks)
    records = [
        {
            "id": f"{document_id}_{index}",
            "doc_id": document_id,
            "source_name": "acme-handbook-2026.md",
            "source_type": "file",
            "scope": "global",
            "session_id": "",
            "user_id": "default",
            "embedding_model": embedding_engine.model_name,
            "chunk_index": index,
            "text": chunk,
            "vector": vector,
        }
        for index, (chunk, vector) in enumerate(zip(chunks, vectors))
    ]
    await asyncio.to_thread(vector_store.add_chunks, records)
    store.update_document_status(document_id, "INDEXED", chunk_count=len(chunks))
