# Implementation Progress Tracker

This document tracks the implementation status of every requirement specified in [`AGENT.md`](./AGENT.md), the core RAG & ReAct system specifications, and the security/robustness audit.

## Summary Matrix

| Category | Total Items | Completed | Status |
| :--- | :---: | :---: | :---: |
| **Core Stack, `uv` Packaging & LangChain Integration** | 7 | 7 | `[100% COMPLETE]` |
| **Core Features (ReAct Agent & In-Chat SSE Ingestion)** | 6 | 6 | `[100% COMPLETE]` |
| **1. UI/UX Design System & Layout Structure** | 9 | 9 | `[100% COMPLETE]` |
| **2. Settings Modal (Tabs 1–5)** | 15 | 15 | `[100% COMPLETE]` |
| **3. Context Management Architecture (Intra & Cross-Session)** | 8 | 8 | `[100% COMPLETE]` |
| **4. High-Scale, Token & Performance Constraints** | 9 | 9 | `[100% COMPLETE]` |
| **5. Security, Reliability & Operational Hardening** | 8 | 8 | `[100% COMPLETE]` |

---

## 0. Core Stack, `uv` Project Management & LangChain Integration

- [x] **Reproducible `uv` Project & Lockfile (`pyproject.toml`, `uv.lock`, `requirements.txt`)** — Managed via `uv` with pinned dependencies in [`uv.lock`](./uv.lock) and exported [`requirements.txt`](./requirements.txt).
- [x] **Vendored `crispembed` Package (`vendor/crispembed`)** — Packaged locally in [`vendor/crispembed`](./vendor/crispembed/pyproject.toml) and linked via `[tool.uv.sources]` in [`pyproject.toml`](./pyproject.toml) so `uv sync` builds and installs `crispembed` deterministically.
- [x] **LangChain ReAct Agent & Tool Calling (`langchain-openai`, `langchain-core`)** — Implemented in [`app/agent.py`](./app/agent.py) using `ChatOpenAI`, `llm.bind_tools([search_knowledge_base])`, `@tool("search_knowledge_base")`, `SystemMessage`, `HumanMessage`, and `AIMessage`, with support for parallel `tool_calls`.
- [x] **LangChain Embeddings & Text Splitting (`Embeddings`, `RecursiveCharacterTextSplitter`)** — [`EmbeddingEngine`](./app/embeddings.py) implements `langchain_core.embeddings.Embeddings`, and [`chunk_text`](./app/ingestion.py) uses `RecursiveCharacterTextSplitter`.
- [x] **Single-Page Frontend (`HTML`, `Tailwind CSS`, `Alpine.js`)** — Implemented in [`static/index.html`](./static/index.html) and [`static/app.js`](./static/app.js).
- [x] **FastAPI Backend** — Implemented in [`app/main.py`](./app/main.py).
- [x] **LanceDB Embedded Vector Store (Hybrid Vector + FTS Search)** — Implemented in [`app/vector_store.py`](./app/vector_store.py) (`VectorStore.hybrid_search`) combining dense vector search + Tantivy BM25 Full-Text Search via Reciprocal Rank Fusion (RRF).

---

## 1. UI/UX & Component Architecture Specification (`AGENT.md` §1–§2)

### 1.1 Design System & Tokens
- [x] **Swiss/Brutalist Minimalist Monochrome Aesthetic** — High-contrast monochrome palette (`#000000`, `#0A0A0A`, `#141414`, `#FFFFFF`, `#888888`, `#444444`, `#222222`) in [`static/index.html`](./static/index.html).
- [x] **Strict `0px` Border Radius (`rounded-none`)** — Enforced globally via CSS (`* { border-radius: 0px !important; }`) and Tailwind config in [`static/index.html`](./static/index.html).
- [x] **Typography (`Inter` + `JetBrains Mono`)** — Configured in [`static/index.html`](./static/index.html) for UI vs. monospace metadata/IDs/code blocks.

### 1.2 Layout Structure
- [x] **Full-Viewport Split Layout (`100vw`, `100vh`, `overflow-hidden`)** — Left sidebar (`260px` fixed, `border-r border-[#222222]`) + Main chat viewport (`flex-1 relative flex flex-col`).
- [x] **Sidebar Header (`New Chat` Button)** — `40px` height, `1px solid #FFFFFF`, inverted hover state (`hover:bg-[#FFFFFF] hover:text-[#000000]`).
- [x] **Sidebar Session List & Two-Step Delete Interaction** — `36px` rows with truncated titles; Click 1 arms `[CONFIRM?]` high-contrast badge with `3000ms` auto-reset and `onMouseLeave` disarm (`handleSessionDeleteClick`, `disarmSessionDelete` in [`static/app.js`](./static/app.js)); Click 2 deletes session and purges session-scoped vectors.
- [x] **Sidebar Footer (`Settings` Trigger)** — Full-width flat button (`44px`) opening the 5-tab `SYSTEM SETTINGS` modal.
- [x] **Centered Message Stream (`max-w-[768px] mx-auto`)** — Distinct `USER` and `ASSISTANT // REACT` monochrome blocks with expandable ReAct traces and citation cards.
- [x] **Floating Prompt Input Container** — Positioned `absolute bottom-6 left-1/2 -translate-x-1/2 w-full max-w-3xl px-4` with auto-expanding `<textarea>` (`48px`–`200px`) and `36px × 36px` submit (`→`) / stop (`■`) button.

---

## 2. Settings Modal Specification (`AGENT.md` §3)

- [x] **Modal Container (`720px × 520px`)** — Centered sharp container with `bg-black/80 backdrop-blur-xs` overlay, `SYSTEM SETTINGS` monospace header, `[X]` close button, and 5 horizontal tabs with `2px solid #FFFFFF` active indicator.

### Tab 1: LLM Configuration
- [x] **Provider-Agnostic Presets** — Predefined OpenAI-compatible providers (`OpenRouter`, `OpenAI`, `xAI (Grok)`, `Google Gemini`, `Groq`, `DeepSeek`, `Together AI`, `Mistral AI`, `Ollama`, `LM Studio`, `Custom`) in [`app/config.py`](./app/config.py) (`PROVIDER_PRESETS`).
- [x] **API Key Input with `SHOW/HIDE` Toggle** — Monospace password input with masked status indicator in [`static/index.html`](./static/index.html).
- [x] **Base URL & Model ID Combobox/Selector** — Editable Base URL and Model ID input paired with provider model suggestions.
- [x] **Parameters (`Temperature` 0.0–2.0 Slider + Numeric Input, `Max Tokens`, `System Prompt`)** — Wired to `PUT /api/settings`.

### Tab 2: Documents (RAG Knowledge Base)
- [x] **Sharp Dashed Dropzone** — Drag-and-drop or click to browse `.PDF`, `.TXT`, `.MD` files (`border-dashed border-[#444444] hover:border-[#FFFFFF]`).
- [x] **Uploaded Files Table** — Displays Filename (monospace), Scope (`GLOBAL` / `SESSION`), File Size & Chunk count, Status Badge (`[INDEXED]`, `[PROCESSING]`, `[FAILED]`), and two-click Delete (`[×]` → `CONFIRM?`).

### Tab 3: Web Sources (URL Scraper/Context)
- [x] **Add URL Bar** — URL input + `ADD` button (`bg-[#FFFFFF] text-[#000000]`).
- [x] **Ingested URLs List** — Displays URL, chunk count, `[SYNCED]` status badge, `↻ SYNC` re-scrape button (`POST /api/documents/{doc_id}/resync`), and two-click Delete button.

### Tab 4: General & Data
- [x] **Retrieval Settings** — CrispEmbed model selector (with automatic LanceDB corpus re-indexing on model switch via `VectorStore.reindex_for_model`), Top-K chunks slider (`1–20`), and Similarity threshold (`0.0–1.0`).
- [x] **High-Contrast Dark / Light Mode Toggle** — Monochrome polarity inversion (`toggleTheme` in [`static/app.js`](./static/app.js)).
- [x] **Data Management** — `Export Chat History (.JSON)` (`GET /api/export`) and `Clear All Sessions` with two-click confirmation (`DELETE /api/sessions`).

### Tab 5: Context & Memory (`AGENT.md` §1.B)
- [x] **Global Memory Ledger** — Lists extracted and manual cross-session facts with individual two-click delete buttons (`[×]`) and `Add Manual Fact` input (`GET/POST/DELETE /api/memories`).
- [x] **Context Window Policy & Sliders** — Selector for `Auto-Summarize`, `Sliding Window`, and `Strict Truncation`, plus `Max Context Tokens` and `Reserved Output Tokens` inputs.
- [x] **Session Reference (`@session` / `/link-session`)** — Validated linking of summarized context from previous sessions into the active chat (`POST /api/sessions/{session_id}/link`).

---

## 3. Context Management Architecture (`AGENT.md` §1.A & §1.B)

- [x] **Token Budget Indicator** — Sharp progress bar and numeric counter (`e.g., 4,120 / 8,192 TOKENS`) in the top-right header (`formattedTokenCounter` in [`static/app.js`](./static/app.js)).
- [x] **Sliding Window & Auto-Summarization (75% Threshold)** — Automatically compresses older turns into `session.summary` after emitting the `done` event when token usage reaches 75% of the context limit (`maybe_compact_session_summary` in [`app/agent.py`](./app/agent.py)).
- [x] **Collapsible `[CONTEXT SUMMARY ACTIVE]` Banner** — Rendered at the top of the message stream allowing users to inspect or edit the rolling summary (`saveEditedSummary` in [`static/app.js`](./static/app.js)).
- [x] **Scoped Attachments (`scope: session_id` vs `scope: global`)** — Files/URLs dropped or uploaded in-chat with session scope are isolated to that session and automatically purged from disk and LanceDB when the session is deleted (`delete_session` in [`app/store.py`](./app/store.py)).
- [x] **Message-Level Context Pruning** — Hover toggle `[EXCLUDE FROM CONTEXT]` / `[INCLUDE IN CONTEXT]` on every message block (`PATCH /api/messages/{message_id}/context`).
- [x] **Floating Input Quick-Toggles (`[GLOBAL DOCS: ON/OFF]`, `[MEMORY: ON/OFF]`)** — Per-prompt opt-in/out pills above the chat textarea.

---

## 4. High-Scale, Token & Performance Constraints (`AGENT.md` §2.A & §2.B)

- [x] **Tiered Context Assembly Pipeline** — Enforces budget allocation (`System Prompt + Global Memory + Linked Sessions + Rolling Summary + RAG Chunks + Recent Turns`) with hard RAG token ceiling (`rag_token_ceiling = 1500`) in [`app/agent.py`](./app/agent.py) (`assemble_tiered_context`, `_format_rag_context_block`).
- [x] **RAG Cosine Deduplication & HTML Boilerplate Stripping** — Deduplicates overlapping chunks via cosine similarity thresholding (`EmbeddingEngine.deduplicate_chunks` in [`app/embeddings.py`](./app/embeddings.py)) and strips `<nav>`, `<header>`, `<footer>`, `<script>`, `<style>`, `<aside>` during URL ingestion (`clean_html_to_text` in [`app/ingestion.py`](./app/ingestion.py)).
- [x] **Prompt Caching Alignment** — Static conversation prefixes (`System Prompt` → `Global Memory` → `Linked Session Context` → `Rolling Summary` → `Prior History`) appear first in outgoing LLM payloads, with per-turn dynamic RAG chunks appended immediately before the new user turn to maximize prefix KV-cache hit rates across turns.
- [x] **Client-Side Heuristic Token Pre-Counting Web Worker** — Runs in [`static/tokenizer.worker.js`](./static/tokenizer.worker.js) using identical heuristic constants (`chars / 3.8`, `words * 1.3`) to [`app/store.py`](./app/store.py) to compute real-time token estimates and block prompts exceeding `hardInputLimit`.
- [x] **Asynchronous SSE Ingestion Pipeline & Lazy FTS Rebuilds** — Non-blocking threadpool execution for PDF/text parsing, CrispEmbed batching, and LanceDB indexing with lazy Tantivy FTS index refreshes (`_ensure_fts_fresh_unlocked` in [`app/vector_store.py`](./app/vector_store.py)) and single end-of-stream document list refresh on the frontend.
- [x] **Multi-Tenant & Session Vector Partitioning** — Filters LanceDB queries by `user_id`, `scope`, `session_id`, and `target_docs` (`_build_where_clause` in [`app/vector_store.py`](./app/vector_store.py)).
- [x] **Streaming & Backpressure (`[STOP GENERATING]` AbortController)** — Wired to `AbortController` on the frontend (`stopGeneration` in [`static/app.js`](./static/app.js)) and `await request.is_disconnected()` on the backend (`run_react_chat_stream` in [`app/agent.py`](./app/agent.py)), preserving partial streamed tokens on mid-stream errors.
- [x] **Frontend Stream Virtualization & LocalStorage Caching** — Windowed message rendering (`visibleMessages`, `virtualLimit` in [`static/app.js`](./static/app.js)) + optimistic `localStorage` caching (`_loadFromLocalCache`, `_saveToLocalCache`).

---

## 5. Security, Reliability & Operational Hardening (`AGENT.md` §2.C)

- [x] **Strict Provider-Isolated AES-256-GCM Credentials** — API keys are encrypted at rest with AES-256-GCM (`encrypt_secret`, `decrypt_secret` in [`app/config.py`](./app/config.py)) and strictly isolated per provider ID (`encrypted_keys[provider]`) in [`app/store.py`](./app/store.py) and `sessionStorage` in [`static/app.js`](./static/app.js) — preventing cross-provider API-key leakage.
- [x] **SSRF Protection on URL Ingestion** — [`validate_url_against_ssrf`](./app/ingestion.py) and [`fetch_url_safely`](./app/ingestion.py) resolve DNS records and block loopback (`127.0.0.0/8`, `::1`), private LAN ranges (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`), link-local cloud metadata (`169.254.169.254`), and non-HTTP(S) schemes on both initial requests and every redirect hop.
- [x] **Bounded Upload Stream (`HTTP 413`)** — [`_read_upload_bounded`](./app/main.py) streams file uploads in 64 KB chunks and aborts immediately with `HTTP 413 Payload Too Large` if `MAX_UPLOAD_BYTES` (default 20 MB) is exceeded.
- [x] **Repository Secret & Data Protection (`.gitignore` & `127.0.0.1` Default Bind)** — [`.gitignore`](./.gitignore) excludes `.venv/`, `data/` (`.secret_key`, `app.db`, `lancedb/`, `uploads/`), and `.env`; [`run.sh`](./run.sh) binds to `127.0.0.1` by default.
- [x] **Rate Limiting with Periodic Bucket Sweep (`HTTP 429` + `Retry-After`)** — Sliding-window [`RateLimiter`](./app/store.py) automatically sweeps stale IP buckets and returns `Retry-After` headers rendered in the UI.
- [x] **Deterministic Ingest & In-Flight Chat Idempotency** — Deterministic file/URL idempotency keys cache and replay `INDEXED`/`SYNCED` SSE events (`save_idempotency` in [`app/ingestion.py`](./app/ingestion.py)), while chat requests enforce a 60s in-flight lock released on stream completion (`release_idempotency` in [`app/agent.py`](./app/agent.py)).
- [x] **Automated Unit & Integration Test Suite (`tests/test_rag_chatbot.py`)** — 17 `pytest` tests verifying SSRF blocking, provider key isolation, 413 upload limits, LangChain splitter & `@tool`, LanceDB RRF hybrid search & deduplication, context pruning, and idempotency (`uv run pytest`).
