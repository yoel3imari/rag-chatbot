function ragApp() {
  return {
    // --- Core State ---
    sessions: [],
    currentSessionId: null,
    currentSession: null,
    messages: [],
    documents: [],
    memories: [],
    providers: [],
    embeddingModels: [],
    settings: {
      provider: "openrouter",
      api_key: "",
      api_key_masked: "",
      has_api_key: false,
      provider_keys_masked: {},
      base_url: "https://openrouter.ai/api/v1",
      model: "openai/gpt-4o-mini",
      temperature: 0.2,
      max_tokens: 2048,
      system_prompt: "",
      embedding_model: "all-MiniLM-L6-v2",
      top_k: 5,
      similarity_threshold: 0.15,
      dedup_threshold: 0.92,
      theme: "dark",
      compaction_mode: "Auto-Summarize",
      max_context_tokens: 8192,
      reserved_output_tokens: 2048,
      rag_token_ceiling: 1500,
      sliding_window_turns: 10,
    },

    // --- UI Controls ---
    settingsOpen: false,
    activeSettingsTab: "llm", // llm | docs | urls | general | memory
    showApiKey: false,
    settingsSavedBanner: false,
    lastSavedNote: "",

    // --- Chat & Prompt Input ---
    promptText: "",
    isStreaming: false,
    abortController: null,
    useGlobalDocs: true,
    useMemory: true,
    targetDocs: [], // list of source_name strings tagged via @
    showInlineUrlInput: false,
    inlineUrlValue: "",
    inlineIngestScope: "session", // "session" or "global"
    settingsUrlValue: "",
    newMemoryFact: "",
    isDraggingOverPrompt: false,
    isDraggingOverDropzone: false,

    // --- Context Summary Banner ---
    summaryExpanded: false,
    editingSummary: false,
    summaryDraft: "",

    // --- @ Mention & /link-session Autocomplete ---
    mentionOpen: false,
    mentionQuery: "",
    mentionItems: [],
    mentionSelectedIndex: 0,
    mentionTriggerStart: -1,

    // --- Two-Step Delete Armed States ---
    armedSessionDelete: null,
    armedDocDelete: null,
    armedMemDelete: null,
    armedClearAll: false,
    _armTimers: {},

    // --- Live SSE Ingestion Toast Queue ---
    ingestToasts: [], // { id, name, source_type, scope, status, progress, message, error }
    rateLimitCountdown: 0,

    // --- Token Pre-Counting Worker State ---
    tokenWorker: null,
    tokenStats: {
      promptTokens: 0,
      activeInputTokens: 0,
      hardInputLimit: 6144,
      maxContextTokens: 8192,
      exceedsHardLimit: false,
    },

    // --- Virtualization Window for Long Message Streams ---
    virtualLimit: 80,

    // --- RAG & ReAct Evaluation State ---
    evalRuns: [],
    activeEvalRun: null,
    isRunningEval: false,
    evalProgress: 0,
    evalStatusMessage: "",
    evalConfig: {
      mode: "full",
      dataset_source: "golden",
      embedding_model: "",
    },
    customEvalJsonText: "",
    evalCustomError: "",
    expandedEvalCases: {},

    get visibleMessages() {
      if (this.messages.length <= this.virtualLimit) {
        return this.messages;
      }
      return this.messages.slice(this.messages.length - this.virtualLimit);
    },

    get hiddenMessageCount() {
      return Math.max(0, this.messages.length - this.virtualLimit);
    },

    get fileDocuments() {
      return this.documents.filter((d) => d.source_type === "file");
    },

    get urlDocuments() {
      return this.documents.filter((d) => d.source_type === "url");
    },

    get currentProviderPreset() {
      return (
        this.providers.find((p) => p.id === this.settings.provider) || {
          id: "custom",
          name: "Custom",
          models: [],
        }
      );
    },

    get formattedTokenCounter() {
      const used = (this.tokenStats.activeInputTokens || 0).toLocaleString();
      const max = (this.settings.max_context_tokens || 8192).toLocaleString();
      return `${used} / ${max} TOKENS`;
    },

    get tokenUsagePercent() {
      const max = this.settings.max_context_tokens || 8192;
      const used = this.tokenStats.activeInputTokens || 0;
      return Math.min(100, Math.round((used / max) * 100));
    },

    async init() {
      // Load cached state from localStorage for instant optimistic paint
      this._loadFromLocalCache();
      this._initTokenWorker();

      await Promise.all([
        this.fetchMeta(),
        this.fetchSettings(),
        this.fetchSessions(),
        this.fetchDocuments(),
        this.fetchMemories(),
        this.fetchEvalRuns(),
      ]);

      if (this.sessions.length > 0 && !this.currentSessionId) {
        await this.selectSession(this.sessions[0].id);
      }
      this.recalculateTokens();
    },

    _loadFromLocalCache() {
      try {
        const cachedSettings = localStorage.getItem("rag_settings_cache");
        if (cachedSettings) {
          this.settings = { ...this.settings, ...JSON.parse(cachedSettings) };
        }
        const cachedSessions = localStorage.getItem("rag_sessions_cache");
        if (cachedSessions) {
          this.sessions = JSON.parse(cachedSessions);
        }
      } catch (_) {}
    },

    _saveToLocalCache() {
      try {
        const safeSettings = { ...this.settings };
        delete safeSettings.api_key;
        localStorage.setItem("rag_settings_cache", JSON.stringify(safeSettings));
        localStorage.setItem("rag_sessions_cache", JSON.stringify(this.sessions));
      } catch (_) {}
    },

    _initTokenWorker() {
      try {
        this.tokenWorker = new Worker("/static/tokenizer.worker.js");
        this.tokenWorker.onmessage = (e) => {
          this.tokenStats = e.data;
        };
      } catch (_) {}
    },

    recalculateTokens() {
      if (!this.tokenWorker) return;
      this.tokenWorker.postMessage({
        promptText: this.promptText,
        systemPrompt: this.settings.system_prompt || "",
        sessionSummary: this.currentSession?.summary || "",
        messages: this.messages.map((m) => ({
          content: m.content,
          tokens: m.tokens,
          included_in_context: m.included_in_context,
        })),
        memories: this.memories.map((m) => ({ fact: m.fact })),
        useMemory: this.useMemory,
        maxContextTokens: this.settings.max_context_tokens || 8192,
        reservedOutputTokens: this.settings.reserved_output_tokens || 2048,
      });
    },

    uuidv4() {
      if (crypto && crypto.randomUUID) return crypto.randomUUID();
      return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
        const r = (Math.random() * 16) | 0;
        const v = c === "x" ? r : (r & 0x3) | 0x8;
        return v.toString(16);
      });
    },

    async _apiFetch(url, options = {}) {
      const resp = await fetch(url, options);
      if (resp.status === 429) {
        const retryAfter = parseInt(resp.headers.get("Retry-After") || "10", 10);
        this.startRateLimitCountdown(retryAfter);
        throw new Error(`HTTP 429 Rate Limit — Retry in ${retryAfter}s`);
      }
      return resp;
    },

    startRateLimitCountdown(seconds) {
      this.rateLimitCountdown = seconds;
      const timer = setInterval(() => {
        this.rateLimitCountdown = Math.max(0, this.rateLimitCountdown - 1);
        if (this.rateLimitCountdown <= 0) clearInterval(timer);
      }, 1000);
    },

    async fetchMeta() {
      const res = await this._apiFetch("/api/meta");
      if (res.ok) {
        const data = await res.json();
        this.providers = data.providers || [];
        this.embeddingModels = data.embedding_models || [];
      }
    },

    async fetchSettings() {
      const res = await this._apiFetch("/api/settings");
      if (res.ok) {
        const data = await res.json();
        this.settings = { ...this.settings, ...data, api_key: "" };
        this._saveToLocalCache();
      }
    },

    onProviderChange() {
      const preset = this.providers.find((p) => p.id === this.settings.provider);
      if (preset) {
        this.settings.base_url = preset.base_url;
        this.settings.model = preset.default_model;
        const savedMask =
          (this.settings.provider_keys_masked || {})[preset.id] || "";
        this.settings.api_key_masked = savedMask;
        this.settings.has_api_key = Boolean(savedMask);
        this.settings.api_key = "";
      }
    },

    async saveSettings() {
      const payload = { ...this.settings };
      if (!payload.api_key || payload.api_key.trim() === "") {
        delete payload.api_key;
      } else {
        try {
          sessionStorage.setItem(
            `rag_client_api_key_${this.settings.provider}`,
            btoa(payload.api_key.trim())
          );
        } catch (_) {}
      }
      const res = await this._apiFetch("/api/settings", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (res.ok) {
        const updated = await res.json();
        // `reindexed_chunks` is per-save metadata, not a persisted setting —
        // strip it so it never enters this.settings / localStorage.
        const reindexed = updated.reindexed_chunks;
        delete updated.reindexed_chunks;
        this.settings = { ...this.settings, ...updated, api_key: "" };
        this.lastSavedNote =
          reindexed > 0
            ? reindexed === 1
              ? "1 CHUNK RE-EMBEDDED"
              : `${reindexed} CHUNKS RE-EMBEDDED`
            : "";
        this._saveToLocalCache();
        this.settingsSavedBanner = true;
        setTimeout(() => {
          this.settingsSavedBanner = false;
        }, 2200);
        this.recalculateTokens();
      }
    },

    toggleTheme() {
      this.settings.theme = this.settings.theme === "light" ? "dark" : "light";
      this.saveSettings();
    },

    // --- Sessions ---
    async fetchSessions() {
      const res = await this._apiFetch("/api/sessions");
      if (res.ok) {
        this.sessions = await res.json();
        this._saveToLocalCache();
      }
    },

    async createNewSession() {
      if (this.isStreaming) this.stopGeneration();
      const res = await this._apiFetch("/api/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: "New Session" }),
      });
      if (res.ok) {
        const created = await res.json();
        this.sessions.unshift(created);
        await this.selectSession(created.id);
      }
    },

    async selectSession(sessionId) {
      if (this.isStreaming) this.stopGeneration();
      this.currentSessionId = sessionId;
      this.virtualLimit = 80;
      const res = await this._apiFetch(`/api/sessions/${sessionId}`);
      if (res.ok) {
        const data = await res.json();
        this.currentSession = data;
        this.messages = (data.messages || []).map((m) => ({
          ...m,
          showTrace: false,
        }));
        this.summaryDraft = data.summary || "";
        await this.fetchDocuments();
        this.recalculateTokens();
        this.scrollToBottom();
      }
    },

    // Two-Step Delete for Session
    handleSessionDeleteClick(sessionId) {
      if (this.armedSessionDelete === sessionId) {
        clearTimeout(this._armTimers[`sess_${sessionId}`]);
        this.armedSessionDelete = null;
        this.executeDeleteSession(sessionId);
      } else {
        this.armedSessionDelete = sessionId;
        clearTimeout(this._armTimers[`sess_${sessionId}`]);
        this._armTimers[`sess_${sessionId}`] = setTimeout(() => {
          if (this.armedSessionDelete === sessionId) {
            this.armedSessionDelete = null;
          }
        }, 3000);
      }
    },

    disarmSessionDelete(sessionId) {
      if (this.armedSessionDelete === sessionId) {
        clearTimeout(this._armTimers[`sess_${sessionId}`]);
        this.armedSessionDelete = null;
      }
    },

    async executeDeleteSession(sessionId) {
      const res = await this._apiFetch(`/api/sessions/${sessionId}`, {
        method: "DELETE",
      });
      if (res.ok) {
        this.sessions = this.sessions.filter((s) => s.id !== sessionId);
        if (this.currentSessionId === sessionId) {
          if (this.sessions.length > 0) {
            await this.selectSession(this.sessions[0].id);
          } else {
            await this.createNewSession();
          }
        }
        await this.fetchDocuments();
      }
    },

    // Two-Step Clear All Sessions
    handleClearAllSessionsClick() {
      if (this.armedClearAll) {
        this.armedClearAll = false;
        this.executeClearAllSessions();
      } else {
        this.armedClearAll = true;
        setTimeout(() => {
          this.armedClearAll = false;
        }, 3000);
      }
    },

    async executeClearAllSessions() {
      const res = await this._apiFetch("/api/sessions", { method: "DELETE" });
      if (res.ok) {
        const data = await res.json();
        await this.fetchSessions();
        if (data.active_session) {
          await this.selectSession(data.active_session.id);
        }
        await this.fetchDocuments();
      }
    },

    async saveEditedSummary() {
      if (!this.currentSessionId) return;
      const res = await this._apiFetch(`/api/sessions/${this.currentSessionId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ summary: this.summaryDraft }),
      });
      if (res.ok) {
        const updated = await res.json();
        this.currentSession.summary = updated.summary;
        this.editingSummary = false;
        this.recalculateTokens();
      }
    },

    async toggleMessageContext(msg) {
      const nextState = !msg.included_in_context;
      msg.included_in_context = nextState;
      this.recalculateTokens();
      const res = await this._apiFetch(`/api/messages/${msg.id}/context`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ included_in_context: nextState }),
      });
      if (res.ok) {
        const updated = await res.json();
        msg.included_in_context = updated.included_in_context;
        this.recalculateTokens();
      }
    },

    async linkOtherSession(targetSessionId) {
      if (!this.currentSessionId || !targetSessionId || targetSessionId === this.currentSessionId)
        return;
      const res = await this._apiFetch(`/api/sessions/${this.currentSessionId}/link`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ target_session_id: targetSessionId }),
      });
      if (res.ok) {
        const updated = await res.json();
        this.currentSession.linked_sessions = updated.linked_sessions || [];
      }
    },

    getSessionTitleById(sid) {
      const s = this.sessions.find((item) => item.id === sid);
      return s ? s.title : sid.slice(0, 8);
    },

    // --- Documents & SSE Ingestion ---
    async fetchDocuments() {
      const q = this.currentSessionId
        ? `?session_id=${encodeURIComponent(this.currentSessionId)}`
        : "";
      const res = await this._apiFetch(`/api/documents${q}`);
      if (res.ok) {
        this.documents = await res.json();
      }
    },

    formatBytes(bytes) {
      const b = Number(bytes) || 0;
      if (b < 1024) return `${b} B`;
      if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
      return `${(b / (1024 * 1024)).toFixed(2)} MB`;
    },

    triggerChatFileUpload() {
      const input = document.getElementById("chat-file-input");
      if (input) input.click();
    },

    triggerSettingsFileUpload() {
      const input = document.getElementById("settings-file-input");
      if (input) input.click();
    },

    async handleFileInputChange(event, scopeMode) {
      const files = event.target.files;
      if (!files || !files.length) return;
      for (const file of files) {
        const scope =
          scopeMode === "session" && this.currentSessionId
            ? this.currentSessionId
            : "global";
        await this.uploadFileWithSSE(file, scope);
      }
      event.target.value = "";
    },

    async handlePromptDrop(event) {
      this.isDraggingOverPrompt = false;
      const files = event.dataTransfer?.files;
      if (!files || !files.length) return;
      for (const file of files) {
        // Scoped Attachments: files dropped into floating prompt input are indexed with scope: session_id
        const scope = this.currentSessionId || "global";
        await this.uploadFileWithSSE(file, scope);
      }
    },

    async handleSettingsDrop(event) {
      this.isDraggingOverDropzone = false;
      const files = event.dataTransfer?.files;
      if (!files || !files.length) return;
      for (const file of files) {
        await this.uploadFileWithSSE(file, "global");
      }
    },

    _upsertToast(toastData) {
      const idx = this.ingestToasts.findIndex((t) => t.id === toastData.id);
      if (idx >= 0) {
        this.ingestToasts.splice(idx, 1, { ...this.ingestToasts[idx], ...toastData });
      } else {
        this.ingestToasts.unshift(toastData);
      }
      if (toastData.status === "INDEXED" || toastData.status === "SYNCED") {
        setTimeout(() => {
          this.ingestToasts = this.ingestToasts.filter((t) => t.id !== toastData.id);
        }, 4200);
      }
    },

    dismissToast(id) {
      this.ingestToasts = this.ingestToasts.filter((t) => t.id !== id);
    },

    async uploadFileWithSSE(file, scope = "global") {
      const tempId = this.uuidv4();
      const idempotencyKey = `file:${file.name}:${file.size}:${file.lastModified || 0}:${scope}`;
      this._upsertToast({
        id: tempId,
        name: file.name,
        source_type: "file",
        scope,
        status: "QUEUED",
        progress: 5,
        message: `Uploading ${file.name}...`,
      });

      const formData = new FormData();
      formData.append("file", file);
      formData.append("scope", scope);
      formData.append("session_id", this.currentSessionId || "");
      formData.append("idempotency_key", idempotencyKey);

      try {
        const resp = await this._apiFetch("/api/ingest/file", {
          method: "POST",
          body: formData,
        });
        if (!resp.ok) {
          const errJson = await resp.json().catch(() => ({}));
          throw new Error(errJson.detail || `Upload failed (${resp.status})`);
        }
        await this._consumeIngestSSEStream(resp, tempId);
      } catch (err) {
        this._upsertToast({
          id: tempId,
          name: file.name,
          status: "FAILED",
          progress: 100,
          message: err.message || "Upload failed",
        });
      }
    },

    async submitUrlWithSSE(urlStr, scope = "global") {
      if (!urlStr || !urlStr.trim()) return;
      const cleanUrl = urlStr.trim();
      const tempId = this.uuidv4();
      const idempotencyKey = `url:${cleanUrl}:${scope}`;
      this._upsertToast({
        id: tempId,
        name: cleanUrl,
        source_type: "url",
        scope,
        status: "QUEUED",
        progress: 5,
        message: `Connecting to ${cleanUrl}...`,
      });

      try {
        const resp = await this._apiFetch("/api/ingest/url", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            url: cleanUrl,
            scope,
            session_id: this.currentSessionId || "",
            idempotency_key: idempotencyKey,
          }),
        });
        if (!resp.ok) {
          const errJson = await resp.json().catch(() => ({}));
          throw new Error(errJson.detail || `URL ingest failed (${resp.status})`);
        }
        await this._consumeIngestSSEStream(resp, tempId);
      } catch (err) {
        this._upsertToast({
          id: tempId,
          name: cleanUrl,
          status: "FAILED",
          progress: 100,
          message: err.message || "URL ingestion failed",
        });
      }
    },

    async submitInlineUrl() {
      if (!this.inlineUrlValue.trim()) return;
      const url = this.inlineUrlValue.trim();
      this.inlineUrlValue = "";
      this.showInlineUrlInput = false;
      const scope =
        this.inlineIngestScope === "session" && this.currentSessionId
          ? this.currentSessionId
          : "global";
      await this.submitUrlWithSSE(url, scope);
    },

    async submitSettingsUrl() {
      if (!this.settingsUrlValue.trim()) return;
      const url = this.settingsUrlValue.trim();
      this.settingsUrlValue = "";
      await this.submitUrlWithSSE(url, "global");
    },

    async resyncUrlDoc(doc) {
      const tempId = doc.id;
      this._upsertToast({
        id: tempId,
        name: doc.name,
        source_type: "url",
        scope: doc.scope,
        status: "QUEUED",
        progress: 10,
        message: `Re-syncing ${doc.name}...`,
      });
      try {
        const resp = await this._apiFetch(`/api/documents/${doc.id}/resync`, {
          method: "POST",
        });
        if (!resp.ok) throw new Error(`Re-sync failed (${resp.status})`);
        await this._consumeIngestSSEStream(resp, tempId);
      } catch (err) {
        this._upsertToast({
          id: tempId,
          name: doc.name,
          status: "FAILED",
          progress: 100,
          message: err.message || "Re-sync failed",
        });
      }
    },

    async _consumeIngestSSEStream(resp, toastId) {
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop() || "";

        for (const block of parts) {
          const line = block.trim();
          if (!line.startsWith("data:")) continue;
          const jsonStr = line.slice(5).trim();
          try {
            const ev = JSON.parse(jsonStr);
            this._upsertToast({
              id: toastId,
              name: ev.name,
              source_type: ev.source_type,
              scope: ev.scope,
              status: ev.status,
              progress: ev.progress || 50,
              message: ev.message,
              error: ev.error,
            });
            if (ev.document && ev.document.id) {
              const existingIdx = this.documents.findIndex(
                (d) => d.id === ev.document.id
              );
              if (existingIdx >= 0) {
                this.documents.splice(existingIdx, 1, ev.document);
              } else {
                this.documents.unshift(ev.document);
              }
            }
          } catch (_) {}
        }
      }
      // Single fetch at end-of-stream to avoid hammering /api/documents
      await this.fetchDocuments();
    },

    // Two-Step Document Delete
    handleDocDeleteClick(docId) {
      if (this.armedDocDelete === docId) {
        clearTimeout(this._armTimers[`doc_${docId}`]);
        this.armedDocDelete = null;
        this.executeDeleteDocument(docId);
      } else {
        this.armedDocDelete = docId;
        clearTimeout(this._armTimers[`doc_${docId}`]);
        this._armTimers[`doc_${docId}`] = setTimeout(() => {
          if (this.armedDocDelete === docId) {
            this.armedDocDelete = null;
          }
        }, 3000);
      }
    },

    disarmDocDelete(docId) {
      if (this.armedDocDelete === docId) {
        clearTimeout(this._armTimers[`doc_${docId}`]);
        this.armedDocDelete = null;
      }
    },

    async executeDeleteDocument(docId) {
      const doc = this.documents.find((d) => d.id === docId);
      const res = await this._apiFetch(`/api/documents/${docId}`, {
        method: "DELETE",
      });
      if (res.ok) {
        this.documents = this.documents.filter((d) => d.id !== docId);
        if (doc) {
          this.targetDocs = this.targetDocs.filter((t) => t !== doc.name);
        }
      }
    },

    // --- Global Memory Ledger ---
    async fetchMemories() {
      const res = await this._apiFetch("/api/memories");
      if (res.ok) {
        this.memories = await res.json();
        this.recalculateTokens();
      }
    },

    async addManualMemory() {
      if (!this.newMemoryFact.trim()) return;
      const res = await this._apiFetch("/api/memories", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          fact: this.newMemoryFact.trim(),
          category: "manual",
        }),
      });
      if (res.ok) {
        this.newMemoryFact = "";
        await this.fetchMemories();
      }
    },

    handleMemDeleteClick(memId) {
      if (this.armedMemDelete === memId) {
        clearTimeout(this._armTimers[`mem_${memId}`]);
        this.armedMemDelete = null;
        this.executeDeleteMemory(memId);
      } else {
        this.armedMemDelete = memId;
        clearTimeout(this._armTimers[`mem_${memId}`]);
        this._armTimers[`mem_${memId}`] = setTimeout(() => {
          if (this.armedMemDelete === memId) {
            this.armedMemDelete = null;
          }
        }, 3000);
      }
    },

    disarmMemDelete(memId) {
      if (this.armedMemDelete === memId) {
        clearTimeout(this._armTimers[`mem_${memId}`]);
        this.armedMemDelete = null;
      }
    },

    async executeDeleteMemory(memId) {
      const res = await this._apiFetch(`/api/memories/${memId}`, {
        method: "DELETE",
      });
      if (res.ok) {
        this.memories = this.memories.filter((m) => m.id !== memId);
        this.recalculateTokens();
      }
    },

    async exportHistoryJson() {
      const res = await this._apiFetch("/api/export");
      if (res.ok) {
        const data = await res.json();
        const blob = new Blob([JSON.stringify(data, null, 2)], {
          type: "application/json",
        });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `rag-chat-history-${new Date().toISOString().slice(0, 10)}.json`;
        a.click();
        URL.revokeObjectURL(url);
      }
    },

    // --- RAG & ReAct Evaluation Suite ---
    openEvaluationTab() {
      this.settingsOpen = true;
      this.activeSettingsTab = "eval";
      this.fetchEvalRuns();
    },

    async fetchEvalRuns() {
      try {
        const res = await this._apiFetch("/api/eval/runs");
        if (res.ok) {
          this.evalRuns = await res.json();
          if (!this.activeEvalRun && this.evalRuns.length > 0) {
            this.activeEvalRun = this.evalRuns[0];
          }
        }
      } catch (_) {}
    },

    async selectEvalRunById(runId) {
      if (!runId) return;
      const found = this.evalRuns.find((r) => r.id === runId);
      if (found) {
        this.activeEvalRun = found;
        this.expandedEvalCases = {};
        return;
      }
      const res = await this._apiFetch(`/api/eval/runs/${runId}`);
      if (res.ok) {
        this.activeEvalRun = await res.json();
        this.expandedEvalCases = {};
      }
    },

    async deleteEvalRun(runId) {
      if (!runId) return;
      const res = await this._apiFetch(`/api/eval/runs/${runId}`, {
        method: "DELETE",
      });
      if (res.ok) {
        this.evalRuns = this.evalRuns.filter((r) => r.id !== runId);
        if (this.activeEvalRun && this.activeEvalRun.id === runId) {
          this.activeEvalRun = this.evalRuns.length > 0 ? this.evalRuns[0] : null;
        }
      }
    },

    toggleEvalCaseExpand(caseId) {
      this.expandedEvalCases = {
        ...this.expandedEvalCases,
        [caseId]: !this.expandedEvalCases[caseId],
      };
    },

    async loadGoldenIntoCustomEditor() {
      this.evalCustomError = "";
      const res = await this._apiFetch("/api/eval/dataset");
      if (res.ok) {
        const data = await res.json();
        this.customEvalJsonText = JSON.stringify(data.cases || [], null, 2);
      }
    },

    async generateSyntheticPreview() {
      this.evalCustomError = "";
      this.evalStatusMessage = "Generating synthetic Q&A test cases from indexed LanceDB chunks...";
      try {
        const res = await this._apiFetch("/api/eval/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ max_cases: 6, use_llm: false }),
        });
        if (res.ok) {
          const data = await res.json();
          this.customEvalJsonText = JSON.stringify(data.cases || [], null, 2);
          this.evalConfig.dataset_source = "custom";
        }
      } catch (err) {
        this.evalCustomError = err.message || "Failed to generate synthetic cases";
      } finally {
        this.evalStatusMessage = "";
      }
    },

    async runEvaluationSuite() {
      if (this.isRunningEval) return;
      this.evalCustomError = "";
      let customCases = null;

      if (this.evalConfig.dataset_source === "custom") {
        try {
          const parsed = JSON.parse(this.customEvalJsonText || "[]");
          customCases = Array.isArray(parsed) ? parsed : parsed.cases || [];
          if (!customCases.length) {
            this.evalCustomError = "Custom JSON dataset must contain at least 1 test case.";
            return;
          }
        } catch (err) {
          this.evalCustomError = `Invalid JSON: ${err.message}`;
          return;
        }
      }

      this.isRunningEval = true;
      this.evalProgress = 5;
      this.evalStatusMessage = "Initializing benchmark suite...";
      this.expandedEvalCases = {};

      const payload = {
        mode: this.evalConfig.mode,
        dataset_source: this.evalConfig.dataset_source,
        custom_cases: customCases,
        embedding_model: this.evalConfig.embedding_model || this.settings.embedding_model,
        top_k: this.settings.top_k,
        similarity_threshold: this.settings.similarity_threshold,
        dedup_threshold: this.settings.dedup_threshold,
      };

      try {
        const resp = await this._apiFetch("/api/eval/run/stream", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        if (!resp.ok) {
          const errJson = await resp.json().catch(() => ({}));
          throw new Error(errJson.detail || `Evaluation failed (${resp.status})`);
        }

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const parts = buffer.split("\n\n");
          buffer = parts.pop() || "";

          for (const block of parts) {
            const line = block.trim();
            if (!line.startsWith("data:")) continue;
            const rawJson = line.slice(5).trim();
            try {
              const ev = JSON.parse(rawJson);
              if (ev.type === "eval_start") {
                this.evalProgress = 10;
                this.evalStatusMessage = `Running ${ev.dataset_name} (${ev.total_cases} cases)...`;
              } else if (ev.type === "case_result") {
                this.evalProgress = ev.progress || 50;
                this.evalStatusMessage = `Evaluated case ${ev.index}/${ev.total_cases}: ${ev.result?.case_id}`;
              } else if (ev.type === "eval_complete" && ev.run) {
                this.evalProgress = 100;
                this.activeEvalRun = ev.run;
                await this.fetchEvalRuns();
              }
            } catch (_) {}
          }
        }
      } catch (err) {
        this.evalCustomError = err.message || "Evaluation run failed";
      } finally {
        this.isRunningEval = false;
        this.evalStatusMessage = "";
      }
    },

    exportActiveEvalJson() {
      if (!this.activeEvalRun) return;
      const blob = new Blob([JSON.stringify(this.activeEvalRun, null, 2)], {
        type: "application/json",
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `rag-eval-${this.activeEvalRun.mode}-${new Date().toISOString().slice(0, 10)}.json`;
      a.click();
      URL.revokeObjectURL(url);
    },

    // --- @ Mention & /link-session Autocomplete in Prompt Input ---
    onPromptInput(event) {
      const el = event.target;
      // Auto-expand textarea (min 48px, max 200px)
      el.style.height = "auto";
      el.style.height = Math.min(200, Math.max(48, el.scrollHeight)) + "px";

      this.recalculateTokens();
      this._checkAutocompleteTrigger(el);
    },

    _checkAutocompleteTrigger(textarea) {
      const cursor = textarea.selectionStart || 0;
      const textBefore = this.promptText.slice(0, cursor);

      // Check for @mention trigger or /link-session command
      const atMatch = textBefore.match(/(?:^|\s)@([^\s]*)$/);
      const slashMatch = textBefore.match(/(?:^|\s)\/link-session(?:\s+([^\s]*))?$/);

      if (slashMatch) {
        const q = (slashMatch[1] || "").toLowerCase();
        this.mentionTriggerStart = textBefore.lastIndexOf("/link-session");
        this.mentionQuery = q;
        this.mentionItems = this.sessions
          .filter(
            (s) =>
              s.id !== this.currentSessionId &&
              s.title.toLowerCase().includes(q)
          )
          .map((s) => ({
            kind: "session",
            id: s.id,
            label: s.title,
            badge: "SESSION",
          }));
        this.mentionOpen = this.mentionItems.length > 0;
        this.mentionSelectedIndex = 0;
        return;
      }

      if (atMatch) {
        const q = (atMatch[1] || "").toLowerCase();
        this.mentionTriggerStart = textBefore.lastIndexOf("@");
        this.mentionQuery = q;

        const docItems = this.documents
          .filter(
            (d) =>
              (d.status === "INDEXED" || d.status === "SYNCED") &&
              d.name.toLowerCase().includes(q)
          )
          .map((d) => ({
            kind: "doc",
            id: d.id,
            label: d.name,
            badge: d.scope === "global" ? `GLOBAL ${d.source_type.toUpperCase()}` : `SESSION ${d.source_type.toUpperCase()}`,
          }));

        const sessItems = this.sessions
          .filter(
            (s) =>
              s.id !== this.currentSessionId &&
              (`session:${s.title}`.toLowerCase().includes(q) || s.title.toLowerCase().includes(q))
          )
          .slice(0, 4)
          .map((s) => ({
            kind: "session",
            id: s.id,
            label: s.title,
            badge: "@SESSION",
          }));

        this.mentionItems = [...docItems, ...sessItems];
        this.mentionOpen = this.mentionItems.length > 0;
        this.mentionSelectedIndex = 0;
      } else {
        this.mentionOpen = false;
      }
    },

    onPromptKeydown(event) {
      if (this.mentionOpen && this.mentionItems.length > 0) {
        if (event.key === "ArrowDown") {
          event.preventDefault();
          this.mentionSelectedIndex =
            (this.mentionSelectedIndex + 1) % this.mentionItems.length;
          return;
        }
        if (event.key === "ArrowUp") {
          event.preventDefault();
          this.mentionSelectedIndex =
            (this.mentionSelectedIndex - 1 + this.mentionItems.length) %
            this.mentionItems.length;
          return;
        }
        if (event.key === "Enter" || event.key === "Tab") {
          event.preventDefault();
          this.selectMentionItem(this.mentionItems[this.mentionSelectedIndex]);
          return;
        }
        if (event.key === "Escape") {
          event.preventDefault();
          this.mentionOpen = false;
          return;
        }
      }

      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        this.sendMessage();
      }
    },

    selectMentionItem(item) {
      if (!item) return;
      const textarea = document.getElementById("chat-prompt-textarea");
      const cursor = textarea ? textarea.selectionStart : this.promptText.length;

      if (item.kind === "doc") {
        if (!this.targetDocs.includes(item.label)) {
          this.targetDocs.push(item.label);
        }
        const before = this.promptText.slice(0, this.mentionTriggerStart);
        const after = this.promptText.slice(cursor);
        this.promptText = `${before}@${item.label} ${after.replace(/^\s+/, "")}`;
      } else if (item.kind === "session") {
        this.linkOtherSession(item.id);
        const before = this.promptText.slice(0, this.mentionTriggerStart);
        const after = this.promptText.slice(cursor);
        this.promptText = `${before}${after}`.trim();
      }

      this.mentionOpen = false;
      this.recalculateTokens();
      this.$nextTick(() => {
        if (textarea) textarea.focus();
      });
    },

    removeTargetDoc(docName) {
      this.targetDocs = this.targetDocs.filter((d) => d !== docName);
    },

    // --- ReAct Chat Submission & SSE Streaming ---
    stopGeneration() {
      if (this.abortController) {
        this.abortController.abort();
        this.abortController = null;
      }
      this.isStreaming = false;
    },

    async sendMessage() {
      if (this.isStreaming) {
        this.stopGeneration();
        return;
      }
      const text = this.promptText.trim();
      if (!text) return;
      if (this.tokenStats.exceedsHardLimit) return;

      if (!this.currentSessionId) {
        await this.createNewSession();
      }

      // Also parse any inline @docName mentions typed directly in promptText
      const activeDocNames = [...this.targetDocs];
      for (const doc of this.documents) {
        if (text.includes(`@${doc.name}`) && !activeDocNames.includes(doc.name)) {
          activeDocNames.push(doc.name);
        }
      }

      this.promptText = "";
      this.mentionOpen = false;
      const textarea = document.getElementById("chat-prompt-textarea");
      if (textarea) textarea.style.height = "48px";

      // Optimistically append user message + assistant placeholder
      const tempUserMsg = {
        id: "temp-user-" + Date.now(),
        role: "user",
        content: text,
        react_steps: [],
        citations: [],
        included_in_context: true,
        tokens: Math.max(1, Math.round(text.length / 4)),
      };
      const assistantPlaceholder = {
        id: "temp-assistant-" + Date.now(),
        role: "assistant",
        content: "",
        react_steps: [],
        citations: [],
        included_in_context: true,
        showTrace: false,
        isStreaming: true,
      };

      this.messages.push(tempUserMsg, assistantPlaceholder);
      const assistantIdx = this.messages.length - 1;
      const userIdx = this.messages.length - 2;

      this.isStreaming = true;
      this.abortController = new AbortController();
      this.scrollToBottom();

      // Retrieve optional client-side sessionStorage key scoped to active provider
      let clientHeaderKey = "";
      try {
        const b64 = sessionStorage.getItem(
          `rag_client_api_key_${this.settings.provider}`
        );
        if (b64) clientHeaderKey = atob(b64);
      } catch (_) {}

      const headers = { "Content-Type": "application/json" };
      if (clientHeaderKey) {
        headers["X-LLM-API-Key"] = clientHeaderKey;
      }

      try {
        const resp = await this._apiFetch("/api/chat/stream", {
          method: "POST",
          headers,
          signal: this.abortController.signal,
          body: JSON.stringify({
            session_id: this.currentSessionId,
            message: text,
            target_docs: activeDocNames,
            use_global_docs: this.useGlobalDocs,
            use_memory: this.useMemory,
            linked_session_ids: this.currentSession?.linked_sessions || [],
            idempotency_key: this.uuidv4(),
            client_token_estimate: this.tokenStats.promptTokens,
          }),
        });

        if (!resp.ok) {
          const errData = await resp.json().catch(() => ({}));
          throw new Error(errData.detail || `Chat error (${resp.status})`);
        }

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const blocks = buffer.split("\n\n");
          buffer = blocks.pop() || "";

          for (const block of blocks) {
            const line = block.trim();
            if (!line.startsWith("data:")) continue;
            const rawJson = line.slice(5).trim();
            try {
              const ev = JSON.parse(rawJson);
              if (ev.type === "session_titled") {
                const s = this.sessions.find((x) => x.id === ev.session_id);
                if (s) s.title = ev.title;
                if (this.currentSession) this.currentSession.title = ev.title;
              } else if (ev.type === "user_message_saved" && ev.message) {
                this.messages[userIdx] = {
                  ...ev.message,
                  showTrace: false,
                };
              } else if (ev.type === "react_step") {
                this.messages[assistantIdx].react_steps.push(ev);
                this.scrollToBottom();
              } else if (ev.type === "citations") {
                this.messages[assistantIdx].citations = ev.citations || [];
                this.scrollToBottom();
              } else if (ev.type === "token") {
                this.messages[assistantIdx].content += ev.delta || "";
                this.scrollToBottom();
              } else if (ev.type === "summary_updated") {
                if (this.currentSession) {
                  this.currentSession.summary = ev.summary;
                  this.summaryDraft = ev.summary;
                }
              } else if (ev.type === "memory_extracted" && ev.memory) {
                await this.fetchMemories();
              } else if (ev.type === "done" && ev.message) {
                this.messages[assistantIdx] = {
                  ...ev.message,
                  showTrace: false,
                  isStreaming: false,
                };
              }
            } catch (_) {}
          }
        }
      } catch (err) {
        if (err.name === "AbortError") {
          this.messages[assistantIdx].content += "\n\n`[GENERATION ABORTED BY USER]`";
        } else if (this.messages[assistantIdx].content) {
          this.messages[assistantIdx].content += `\n\n⚠️ **Error:** ${err.message}`;
        } else {
          this.messages[assistantIdx].content = `⚠️ **Error:** ${err.message}`;
        }
        this.messages[assistantIdx].isStreaming = false;
      } finally {
        this.isStreaming = false;
        this.abortController = null;
        this.recalculateTokens();
        this.scrollToBottom();
      }
    },

    scrollToBottom() {
      this.$nextTick(() => {
        const stream = document.getElementById("chat-message-stream");
        if (stream) {
          stream.scrollTop = stream.scrollHeight;
        }
      });
    },

    renderMarkdown(raw) {
      if (!raw) return "";
      let html = String(raw)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");

      // Code blocks ```lang ... ```
      html = html.replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g, (_, lang, code) => {
        const badge = lang ? `<div class="text-xs font-mono uppercase tracking-wider px-3.5 py-1.5 border-b border-[#222222] text-[#888888]">${lang}</div>` : "";
        return `<div class="my-3.5 border border-[#333333] bg-[#050505] overflow-x-auto">${badge}<pre class="p-3.5 text-sm font-mono leading-relaxed"><code>${code.trim()}</code></pre></div>`;
      });

      // Inline code
      html = html.replace(
        /`([^`]+)`/g,
        '<code class="px-1.5 py-0.5 text-sm font-mono border border-[#333333] bg-[#141414]">$1</code>'
      );

      // Bold & strict Italic (requiring non-space adjacent to asterisks so 2 * 3 * 4 is untouched)
      html = html.replace(/\*\*([^*]+)\*\*/g, '<strong class="font-semibold">$1</strong>');
      html = html.replace(
        /(^|[\s(])\*([^\s*][^*]*?)\*(?=[\s).,!?:;]|$)/g,
        '$1<em class="italic text-[#BBBBBB]">$2</em>'
      );

      // Blockquotes
      html = html.replace(
        /^&gt;\s?(.*)$/gm,
        '<blockquote class="border-l-2 border-white pl-3.5 py-0.5 my-2 text-[#AAAAAA]">$1</blockquote>'
      );

      // Citations in brackets e.g. [acme-handbook-2026.md]
      html = html.replace(
        /\[([a-zA-Z0-9_.\-/]+\.(?:pdf|txt|md|html)|https?:\/\/[^\]]+)\]/gi,
        '<span class="inline-block px-1.5 py-0.5 text-xs font-mono border border-white bg-white text-black font-medium">[$1]</span>'
      );

      // Paragraphs / line breaks
      html = html.replace(/\n\n/g, "</p><p class='mt-3'>");
      html = html.replace(/\n/g, "<br>");
      return `<p>${html}</p>`;
    },
  };
}
