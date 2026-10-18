### UI/UX & Component Architecture Specification

#### 1. Design System & Tokens

* **Aesthetic:** Minimalist, high-contrast monochrome (Swiss/Brutalist modern).
* **Geometry:** Strictly `0px` border-radius (`rounded-none`) across all elements—buttons, inputs, modals, cards, and scrollbars.
* **Color Palette:**
* **Backgrounds:** Primary `#000000` (or `#FFFFFF` for light mode), Secondary/Surface `#0A0A0A`, Elevated/Hover `#141414`.
* **Text:** Primary `#FFFFFF`, Secondary/Muted `#888888`, Disabled `#444444`.
* **Borders:** `1px solid #222222` (Default), `1px solid #FFFFFF` (Active/Focus).
* **Accents:** Pure inverted contrast (White background with black text for primary actions). Destructive actions use high-contrast inversion or a stark monochrome warning state.


* **Typography:** Clean Neo-Grotesque sans-serif (`Inter` or `Geist`) for UI; monospace (`JetBrains Mono` or `Geist Mono`) for Model IDs, URLs, file metadata, and code blocks.

---

#### 2. Layout Structure

A full-viewport application layout (`100vw`, `100vh`, `overflow-hidden`) divided into two primary panes:

##### A. Left Sidebar (`width: 260px`, fixed, `border-right: 1px solid #222222`)

1. **Header Area:**
* **New Chat Button:** Full-width button (`height: 40px`), `1px solid #FFFFFF`, inverted hover state (`bg-white text-black`), featuring a `+` icon and `"New Chat"` label.


2. **Session List (Scrollable `flex-1`):**
* Vertical stack of session items (`height: 36px`, `px-3`).
* **Text:** Truncated session title (`text-ellipsis overflow-hidden whitespace-nowrap`).
* **Two-Step Delete Interaction:**
* *Default/Hover:* Displays a subtle trash icon on row hover.
* *Click 1 (Armed State):* Icon switches to a checkmark or `"Confirm?"` badge with a high-contrast border (`border-white bg-white text-black`). Automatically resets to default after `3000ms` if unclicked, or on `onMouseLeave`.
* *Click 2 (Execution):* Immediately removes the session from the list.




3. **Footer Area (`border-top: 1px solid #222222`):**
* **Settings Trigger:** Full-width flat button (`height: 44px`), aligned left with a gear/sliders icon and `"Settings"` label.



##### B. Main Chat Viewport (`flex-1`, `relative`, `flex flex-col`)

1. **Message Stream (`flex-1 overflow-y-auto pb-32`):**
* Centered content column (`max-width: 768px`, `mx-auto`).
* Distinct visual separation between `User` and `Assistant` blocks using subtle borders or left-aligned monochrome avatars/badges.


2. **Floating Prompt Input Container:**
* **Positioning:** `absolute bottom-6 left-1/2 -translate-x-1/2 w-full max-w-3xl px-4`.
* **Wrapper:** Sharp-edged box (`bg-[#0A0A0A]`, `border: 1px solid #333333`, focus-within `border-white`, subtle drop shadow `0 10px 30px rgba(0,0,0,0.8)`).
* **Input Field:** Auto-expanding `<textarea>` (`min-height: 48px`, `max-height: 200px`), no border, transparent background.
* **Submit Button:** Positioned bottom-right inside the input box (`width: 36px`, `height: 36px`). Uses a sharp right-arrow icon (`→`). Disabled when input is empty (`opacity-30`); solid white background with black arrow when active.



---

#### 3. Settings Modal Specification

* **Overlay:** Full-screen backdrop (`bg-black/80 backdrop-blur-xs`).
* **Container:** Centered sharp container (`width: 720px`, `height: 520px`, `bg-[#000000]`, `border: 1px solid #333333`, `flex flex-col`).
* **Modal Header:** Title (`"SYSTEM SETTINGS"`) in monospace uppercase + top-right close button (`[X]`).
* **Navigation:** Horizontal tab bar (`border-bottom: 1px solid #222222`) or left vertical sub-nav (`width: 180px`). Active tab indicated by a `2px solid #FFFFFF` bottom/left bar.

##### Tab 1: LLM Configuration

* **API Key:** Password input field with a toggleable visibility icon (`Show/Hide`) and monospace font.
* **Base URL:** Text input with placeholder `[https://api.openai.com/v1](https://api.openai.com/v1)` or local endpoint `http://localhost:11434/v1`.
* **Model ID:** Text input or combobox (e.g., `gpt-4o`, `claude-3-5-sonnet`, `llama3.1:8b`).
* **Parameters:**
* *Temperature:* Numeric input + sharp-edged range slider (`0.0` to `2.0`).
* *Max Tokens:* Numeric input.
* *System Prompt:* Multi-line `<textarea>` for custom base instructions.



##### Tab 2: Documents (RAG Knowledge Base)

* **Dropzone:** Sharp-edged dashed border box (`border: 1px dashed #444444`, hover `border-white`, `p-6`, centered text: `"DRAG & DROP FILES OR CLICK TO BROWSE (.PDF, .TXT, .MD)"`).
* **Uploaded Files List:**
* Table or stacked list with `1px solid #222222` dividers.
* **Columns:** Filename (monospace), File Size (`KB/MB`), Status Badge (`[INDEXED]`, `[PROCESSING]`), and a two-click Delete button (`[×]`).



##### Tab 3: Web Sources (URL Scraper/Context)

* **Add URL Bar:** Flex row containing a URL text input (`flex-1`, placeholder `[https://example.com/docs](https://example.com/docs)`) + `"ADD"` button (`bg-white text-black px-4`).
* **URL List:**
* Stacked list of ingested URLs.
* Displays full/truncated URL, ingestion timestamp or status (`[SYNCED]`), a `"Re-sync"` icon button, and a two-click Delete button.



##### Tab 4: General & Data (Related Settings)

* **Retrieval Settings:** Top-K chunks slider (`1–20`), Similarity threshold input (`0.0–1.0`).
* **Appearance:** High-contrast Dark / Light mode toggle (strictly inverting `#000000` and `#FFFFFF`).
* **Data Management:**
* `"Export Chat History (.JSON)"` button.
* `"Clear All Sessions"` button (requires two-click confirmation).


### Context Management, Scale & Optimization Specification

#### 1. Context Management Architecture (UI & State)

##### A. Intra-Session Context (Single Chat Scope)

* **Token Budget Indicator:** A minimalist, sharp-edged progress bar and numeric counter (`e.g., 4,120 / 8,192 TOKENS`) positioned in the top-right header or pinned inside the floating prompt container's top border.
* **Sliding Window & Summarization Strategy:**
* Maintain the last `N` messages verbatim in the active payload.
* Automatically compress older messages into a rolling `Session Summary` block when the token threshold reaches 75% of the configured context limit.
* **UI Control:** Display a collapsible `[CONTEXT SUMMARY ACTIVE]` banner at the top of the message stream allowing users to inspect or edit the compressed state.


* **Scoped Attachments (Session-Only RAG):**
* Files or URLs dropped directly into the floating prompt input are indexed with `scope: session_id` metadata—isolated from global knowledge bases and automatically purged when the session is deleted.


* **Message-Level Context Pruning:**
* Each message block includes a subtle hover toggle (`[INCLUDE / EXCLUDE FROM CONTEXT]`) to let users manually drop noisy turns or failed generations from subsequent prompt payloads without deleting them from the UI history.



##### B. Cross-Session Context (Global Memory & Knowledge Scope)

* **Two-Tier Knowledge Isolation:**
* **Global Knowledge Base:** Documents and URLs uploaded via the Settings Modal are tagged `scope: global` (or `workspace_id`) and remain retrievable across all sessions.
* **Cross-Session Memory Ledger:** Extracted user facts, preferences, and recurring instructions persisted across chats.


* **UI Controls for Cross-Session Scope:**
* **Floating Input Quick-Toggles:** Sharp-edged monochrome pills (`[GLOBAL DOCS: ON]`, `[MEMORY: ON]`) above the textarea allowing instant per-prompt opting in/out of cross-session retrieval.
* **Settings Modal → Tab 5: Context & Memory:**
* **Global Memory Ledger:** A structured list of extracted cross-session facts/constraints with individual two-click delete buttons (`[×]`) and an `"Add Manual Fact"` input.
* **Context Window Policy:** Selector for compaction mode (`Sliding Window`, `Auto-Summarize`, `Strict Truncation`) and token budget sliders (`Max Context Tokens`, `Reserved Output Tokens`).
* **Session Reference (`@session`):** Ability to reference or import summarized context from a previous session into the active chat via a slash command (`/link-session`).


---

#### 2. High-Scale, Token & Performance Constraints

##### A. Token Optimization Constraints

1. **Tiered Context Assembly Pipeline:**
* Construct prompt payloads strictly by budget allocation:

$$\text{Total Budget} = \text{System Prompt} + \text{Global Memory} + \text{RAG Chunks} + \text{Rolling Summary} + \text{Recent Turns}$$


* Enforce hard token ceilings per segment (e.g., max `1,500` tokens for RAG context) to prevent document dumps from evicting conversation history.


2. **RAG Deduplication & Reranking:**
* Deduplicate overlapping document/URL chunks via cosine similarity thresholding before injection.
* Strip boilerplate HTML, navigation trees, and redundant whitespace during URL ingestion to maximize information density per token.


3. **Prompt Caching Alignment:**
* Structure outgoing LLM payloads so static prefixes (`System Prompt` -> `Global Rules` -> `Static RAG Context`) appear first, maximizing KV-cache / prefix-caching hit rates on supported endpoints.


4. **Client-Side Pre-Counting:**
* Run a lightweight tokenizer in a Web Worker to calculate real-time prompt token estimates before dispatch, blocking submissions that exceed hard limits.



##### B. High-Concurrency & Multi-Tenant Scale Constraints

1. **Stateless API & Externalized State:**
* Backend workers remain strictly stateless. Active session buffers and rate-limit counters reside in an in-memory store (e.g., Redis), while persistent history and metadata live in a partitioned relational/document store.


2. **Asynchronous Ingestion Pipeline:**
* Document parsing, chunking, embedding generation, and URL scraping execute asynchronously via task queues. The UI reflects real-time state transitions (`[QUEUED]` -> `[PARSING]` -> `[INDEXED]` / `[FAILED]`) via SSE or lightweight polling—never blocking the main thread.


3. **Multi-Tenant Vector Partitioning:**
* Strictly isolate vector embeddings by `user_id` and `session_id` using tenant namespaces or payload metadata filters to prevent cross-user data leakage and avoid full-index scans under high concurrency.


4. **Streaming & Backpressure Handling:**
* Stream LLM responses via Server-Sent Events (SSE) with an explicit `[STOP GENERATING]` abort controller (`AbortSignal`) wired to the floating input button (`■`), immediately terminating upstream LLM token generation on disconnect or cancellation.


5. **Frontend Virtualization & Local Caching:**
* Virtualize the sidebar session list and long chat message streams to maintain 60fps scrolling regardless of history size.
* Cache session lists and static settings locally (`IndexedDB` / `localStorage`) with optimistic UI updates and stale-while-revalidate background syncing.



##### C. Security, Reliability & Operational Constraints

* **Credential Handling:** API keys entered in the UI must either be stored strictly client-side (encrypted in local storage and passed via headers) or encrypted at rest (AES-256-GCM) on the backend—never logged or returned in plaintext API responses.
* **Rate Limiting & Circuit Breaking:** Enforce per-user and per-IP token/request rate limits (`HTTP 429` with `Retry-After` countdown rendered in the UI) and exponential backoff for upstream LLM provider timeouts.
* **Idempotency:** Attach a client-generated `idempotency_key` (UUIDv4) to every prompt submission and document upload to prevent duplicate execution on network retries.
