import tempfile
import time
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.agent import assemble_tiered_context, search_knowledge_base, to_langchain_messages
from app.embeddings import EmbeddingEngine, embedding_engine
from app.ingestion import SSRFBlockedError, chunk_text, clean_html_to_text, validate_url_against_ssrf
from app.main import app
from app.store import AppStore, RateLimiter, estimate_tokens
from app.vector_store import VectorStore


def test_token_estimation_alignment():
    text = "Company vacation policy allows 25 days of paid time off per year."
    tokens = estimate_tokens(text)
    # Verify exact formula: round((len/3.8 + words*1.3) / 2)
    expected = max(1, round((len(text) / 3.8 + len(text.split()) * 1.3) / 2))
    assert tokens == expected


def test_langchain_splitter_and_html_cleaning():
    html = """
    <html>
      <head><title>Engineering Guide</title><script>alert(1)</script></head>
      <body>
        <nav>Home | Pricing | Login</nav>
        <main>
          <h1>Deployment Protocol</h1>
          <p>All production deployments require automated CI checks and zero-downtime rollout.</p>
        </main>
        <footer>Copyright 2026</footer>
      </body>
    </html>
    """
    cleaned = clean_html_to_text(html)
    assert "Deployment Protocol" in cleaned
    assert "zero-downtime rollout" in cleaned
    assert "alert(1)" not in cleaned
    assert "Home | Pricing" not in cleaned
    assert "Copyright 2026" not in cleaned

    long_doc = "\n\n".join([f"Paragraph {i}: " + ("word " * 80) for i in range(6)])
    chunks = chunk_text(long_doc, chunk_size=400, chunk_overlap=60)
    assert len(chunks) >= 4
    assert all(len(c) <= 450 for c in chunks)


@pytest.mark.parametrize(
    "blocked_url",
    [
        "http://localhost:8000/api/settings",
        "http://127.0.0.1/admin",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.1/internal",
        "http://192.168.1.50/router",
        "http://172.16.0.5/secret",
        "http://[::1]:8000/",
        "file:///etc/passwd",
        "ftp://example.com/file.txt",
        "http://service.internal/config",
    ],
)
def test_ssrf_blocklist(blocked_url: str):
    with pytest.raises(SSRFBlockedError):
        validate_url_against_ssrf(blocked_url)


def test_provider_api_key_isolation():
    with tempfile.TemporaryDirectory() as tmp:
        test_store = AppStore(db_path=f"{tmp}/test.db")
        # Save key under openrouter
        test_store.update_settings(
            {"provider": "openrouter", "api_key": "sk-or-secret-key-12345"}
        )
        s_or = test_store.get_settings(include_decrypted_key=True)
        assert s_or["has_api_key"] is True
        assert s_or["_decrypted_api_key"] == "sk-or-secret-key-12345"
        assert s_or["api_key_masked"] == "sk-o••••2345"

        # Switch to groq without providing a groq key -> must NOT leak openrouter key
        test_store.update_settings({"provider": "groq"})
        s_groq = test_store.get_settings(include_decrypted_key=True)
        assert s_groq["provider"] == "groq"
        assert s_groq["_decrypted_api_key"] == ""
        assert s_groq["has_api_key"] is False

        # Switch to custom -> must NOT leak openrouter key
        test_store.update_settings({"provider": "custom", "base_url": "http://example.com/v1"})
        s_custom = test_store.get_settings(include_decrypted_key=True)
        assert s_custom["_decrypted_api_key"] == ""

        # Switch back to openrouter -> openrouter key is still preserved
        test_store.update_settings({"provider": "openrouter"})
        s_or_again = test_store.get_settings(include_decrypted_key=True)
        assert s_or_again["_decrypted_api_key"] == "sk-or-secret-key-12345"
        test_store.close()


def test_rate_limiter_sweep_and_idempotency_ttl():
    limiter = RateLimiter(max_requests_per_minute=2)
    ok1, _ = limiter.check("198.51.100.1")
    ok2, _ = limiter.check("198.51.100.1")
    ok3, retry = limiter.check("198.51.100.1")
    assert ok1 and ok2 and not ok3
    assert retry >= 1

    # Force sweep of stale buckets
    limiter._buckets["198.51.100.99"] = [time.time() - 120.0]
    limiter._last_sweep = time.time() - 120.0
    limiter.check("198.51.100.2")
    assert "198.51.100.99" not in limiter._buckets

    with tempfile.TemporaryDirectory() as tmp:
        test_store = AppStore(db_path=f"{tmp}/idem.db")
        test_store.save_idempotency("k1", "cached_val", ttl_seconds=0.05)
        assert test_store.check_idempotency("k1") == "cached_val"
        time.sleep(0.08)
        assert test_store.check_idempotency("k1") is None

        test_store.save_idempotency("k2", "in_flight", ttl_seconds=60.0)
        assert test_store.check_idempotency("k2") == "in_flight"
        test_store.release_idempotency("k2")
        assert test_store.check_idempotency("k2") is None
        test_store.close()


def test_lancedb_hybrid_search_dedup_and_reindex():
    with tempfile.TemporaryDirectory() as tmp:
        vs = VectorStore(uri=tmp)
        texts = [
            "Acme Corp engineers receive 25 days of paid time off and a 4-week sabbatical.",
            "Acme Corp engineers receive 25 days of paid time off and a 4-week sabbatical!",  # near-duplicate
            "Production deployments are frozen on Fridays after 15:00 UTC.",
        ]
        vecs = embedding_engine.embed_documents(texts)
        records = [
            {
                "id": f"doc1_{i}",
                "doc_id": "doc1",
                "source_name": "handbook.md",
                "source_type": "file",
                "scope": "global",
                "session_id": "",
                "user_id": "default",
                "embedding_model": embedding_engine.model_name,
                "chunk_index": i,
                "text": t,
                "vector": v,
            }
            for i, (t, v) in enumerate(zip(texts, vecs))
        ]
        vs.add_chunks(records)

        # Hybrid search with cosine deduplication (near-duplicate should be collapsed)
        hits = vs.hybrid_search(
            query="How many days of paid time off and sabbatical?",
            target_docs=["handbook.md"],
            top_k=5,
            dedup_threshold=0.90,
        )
        assert len(hits) >= 1
        pto_hits = [h for h in hits if "sabbatical" in h["text"]]
        assert len(pto_hits) == 1  # Deduplicated!

        # Test target_docs filter excluding handbook.md
        empty_hits = vs.hybrid_search(
            query="paid time off",
            target_docs=["nonexistent.pdf"],
            top_k=5,
        )
        assert len(empty_hits) == 0


def test_tiered_context_budget_and_pruning():
    with tempfile.TemporaryDirectory() as tmp:
        test_store = AppStore(db_path=f"{tmp}/ctx.db")
        sess = test_store.create_session(title="Budget Session")
        sid = sess["id"]
        m1 = test_store.add_message(sid, "user", "Keep this message in context.")
        m2 = test_store.add_message(sid, "assistant", "Prune this noisy response.")
        test_store.toggle_message_context(m2["id"], False)
        test_store.update_session(sid, summary="Prior compressed summary fact.")

        full_sess = test_store.get_session(sid)
        settings = test_store.get_settings()
        assembled, budget = assemble_tiered_context(full_sess, settings, use_memory=False)
        lc_msgs = to_langchain_messages(assembled)

        contents = [m.content for m in lc_msgs]
        assert any("Prior compressed summary fact." in c for c in contents)
        assert any("Keep this message in context." in c for c in contents)
        assert not any("Prune this noisy response." in c for c in contents)
        assert budget["summary_tokens"] > 0
        test_store.close()


def test_api_upload_size_limit_link_validation_and_ingest_idempotency(monkeypatch):
    import app.main as main_mod

    monkeypatch.setattr(main_mod, "MAX_UPLOAD_BYTES", 256)

    with TestClient(app) as client:
        # 1. Oversized file upload -> HTTP 413
        oversized = b"x" * 600
        r_413 = client.post(
            "/api/ingest/file",
            files={"file": ("huge.txt", oversized, "text/plain")},
            data={"scope": "global"},
        )
        assert r_413.status_code == 413

        # 2. Valid small file upload with idempotency_key -> caches SSE result
        small_md = b"# Security Policy\nAll keys use AES-256-GCM."
        idem_key = "test-file-idem-key-001"
        r_ok1 = client.post(
            "/api/ingest/file",
            files={"file": ("sec.md", small_md, "text/markdown")},
            data={"scope": "global", "idempotency_key": idem_key},
        )
        assert r_ok1.status_code == 200
        assert "INDEXED" in r_ok1.text
        docs_after_first = len(client.get("/api/documents").json())

        # Replay identical idempotency_key -> returns cached completion without adding duplicate doc
        r_ok2 = client.post(
            "/api/ingest/file",
            files={"file": ("sec.md", small_md, "text/markdown")},
            data={"scope": "global", "idempotency_key": idem_key},
        )
        assert r_ok2.status_code == 200
        assert "INDEXED" in r_ok2.text
        docs_after_second = len(client.get("/api/documents").json())
        assert docs_after_second == docs_after_first

        # 3. Validate link_session rejects nonexistent target session with 404
        s1 = client.post("/api/sessions", json={"title": "Session A"}).json()
        r_link_bad = client.post(
            f"/api/sessions/{s1['id']}/link",
            json={"target_session_id": "nonexistent-session-id"},
        )
        assert r_link_bad.status_code == 404

        s2 = client.post("/api/sessions", json={"title": "Session B"}).json()
        r_link_ok = client.post(
            f"/api/sessions/{s1['id']}/link",
            json={"target_session_id": s2["id"]},
        )
        assert r_link_ok.status_code == 200
        assert s2["id"] in r_link_ok.json()["linked_sessions"]

        # 4. Verify LangChain @tool invocation
        tool_out = search_knowledge_base.invoke(
            {"reasoning": "Check security policy", "query": "AES-256-GCM", "target_source": "sec.md"}
        )
        assert "AES-256-GCM" in tool_out
