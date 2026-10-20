"""Golden evaluation dataset and corpus seeding."""

from typing import Any, Dict, List

from app.embeddings import embedding_engine
from app.ingestion import chunk_text
from app.sample_data import SAMPLE_COMPANY_HANDBOOK
from app.store import store
from app.vector_store import vector_store

SAMPLE_COMPANY_HANDBOOK_TEXT = SAMPLE_COMPANY_HANDBOOK

GOLDEN_EVAL_DATASET: List[Dict[str, Any]] = [
    {
        "id": "golden-01",
        "category": "rag_factual",
        "question": "What are Acme Corp's core synchronous collaboration hours and annual employee stipends?",
        "expected_answer": (
            "Acme Corp's core synchronous collaboration hours are 13:00 UTC to 17:00 UTC Monday through Friday. "
            "Full-time employees receive an annual $2,500 Home Office Stipend and a $1,200 Learning & Conference budget [acme-handbook-2026.md]."
        ),
        "expected_sources": ["acme-handbook-2026.md"],
        "expected_facts": ["13:00 UTC", "17:00 UTC", "$2,500", "$1,200"],
        "target_docs": [],
        "expect_retrieval": True,
        "allow_abstain": False,
    },
    {
        "id": "golden-02",
        "category": "rag_factual",
        "question": "How many days of Paid Time Off (PTO) do employees accrue, and when do engineers qualify for a sabbatical?",
        "expected_answer": (
            "Employees accrue 25 days of Paid Time Off (PTO) per calendar year plus 12 company holidays. "
            "After 4 years of continuous tenure, engineers qualify for a 4-week paid sabbatical [acme-handbook-2026.md]."
        ),
        "expected_sources": ["acme-handbook-2026.md"],
        "expected_facts": ["25 days", "12 company holidays", "4 years", "4-week"],
        "target_docs": [],
        "expect_retrieval": True,
        "allow_abstain": False,
    },
    {
        "id": "golden-03",
        "category": "rag_factual",
        "question": "What is the Friday production deployment freeze rule and the SEV-1 outage response SLA?",
        "expected_answer": (
            "Routine production deployments are frozen on Fridays after 15:00 UTC without VP of Engineering sign-off. "
            "SEV-1 critical outages require initial on-call acknowledgement via PagerDuty within 5 minutes and a public status page update within 15 minutes [acme-handbook-2026.md]."
        ),
        "expected_sources": ["acme-handbook-2026.md"],
        "expected_facts": ["15:00 UTC", "VP of Engineering", "5 minutes", "15 minutes"],
        "target_docs": [],
        "expect_retrieval": True,
        "allow_abstain": False,
    },
    {
        "id": "golden-04",
        "category": "rag_factual",
        "question": "Where and within what timeframe must blameless incident postmortems be published?",
        "expected_answer": (
            "Blameless postmortems must be published in the `/incidents` repository within 48 hours of resolution [acme-handbook-2026.md]."
        ),
        "expected_sources": ["acme-handbook-2026.md"],
        "expected_facts": ["/incidents", "48 hours"],
        "target_docs": [],
        "expect_retrieval": True,
        "allow_abstain": False,
    },
    {
        "id": "golden-05",
        "category": "targeted_doc",
        "question": "What encryption standard and service token rotation schedule are required for production secrets?",
        "expected_answer": (
            "All API keys and production secrets must be encrypted at rest using AES-256-GCM or stored in Vault, "
            "and rotating service tokens every 90 days is mandatory [acme-handbook-2026.md]."
        ),
        "expected_sources": ["acme-handbook-2026.md"],
        "expected_facts": ["AES-256-GCM", "Vault", "90 days"],
        "target_docs": ["acme-handbook-2026.md"],
        "expect_retrieval": True,
        "allow_abstain": False,
    },
    {
        "id": "golden-06",
        "category": "unanswerable",
        "question": "What is Acme Corp's corporate helicopter reimbursement limit and pet insurance deductible?",
        "expected_answer": (
            "The knowledge base does not contain any policy or information regarding corporate helicopter reimbursement or pet insurance deductibles."
        ),
        "expected_sources": [],
        "expected_facts": [],
        "target_docs": [],
        "expect_retrieval": True,
        "allow_abstain": True,
    },
    {
        "id": "golden-07",
        "category": "chitchat",
        "question": "Hello! Good morning!",
        "expected_answer": "Hello! How can I help you today?",
        "expected_sources": [],
        "expected_facts": [],
        "target_docs": [],
        "expect_retrieval": False,
        "allow_abstain": False,
    },
    {
        "id": "golden-08",
        "category": "math",
        "question": "25 * 4 + 50 = ?",
        "expected_answer": "150",
        "expected_sources": [],
        "expected_facts": ["150"],
        "target_docs": [],
        "expect_retrieval": False,
        "allow_abstain": False,
    },
]


def ensure_default_handbook_seeded() -> None:
    """Ensures the default Acme Corp handbook is indexed in SQLite and LanceDB."""
    documents = store.list_documents()
    has_handbook = any(
        document.get("name") == "acme-handbook-2026.md" for document in documents
    )
    try:
        row_count = vector_store.table.count_rows()
    except Exception:
        row_count = 0

    if has_handbook and row_count != 0:
        return

    document = store.create_document_record(
        name="acme-handbook-2026.md",
        source_type="file",
        scope="global",
        file_size=len(SAMPLE_COMPANY_HANDBOOK.encode("utf-8")),
    )
    document_id = document["id"]
    chunks = chunk_text(SAMPLE_COMPANY_HANDBOOK)
    vectors = embedding_engine.embed_batch(chunks)
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
    vector_store.add_chunks(records)
    store.update_document_status(document_id, "INDEXED", chunk_count=len(chunks))
