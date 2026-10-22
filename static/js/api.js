window.RagModules = window.RagModules || {};

window.RagModules.api = {
  async init() {
    this._loadFromLocalCache();
    this.applyTheme();
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
      this.tokenWorker.onmessage = (event) => {
        this.tokenStats = event.data;
      };
    } catch (_) {}
  },

  recalculateTokens() {
    if (!this.tokenWorker) return;
    this.tokenWorker.postMessage({
      promptText: this.promptText,
      systemPrompt: this.settings.system_prompt || "",
      sessionSummary: this.currentSession?.summary || "",
      messages: this.messages.map((message) => ({
        content: message.content,
        tokens: message.tokens,
        included_in_context: message.included_in_context,
      })),
      memories: this.memories.map((memory) => ({ fact: memory.fact })),
      useMemory: this.useMemory,
      maxContextTokens: this.settings.max_context_tokens || 8192,
      reservedOutputTokens: this.settings.reserved_output_tokens || 2048,
    });
  },

  uuidv4() {
    if (crypto && crypto.randomUUID) return crypto.randomUUID();
    return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (character) => {
      const random = (Math.random() * 16) | 0;
      const value = character === "x" ? random : (random & 0x3) | 0x8;
      return value.toString(16);
    });
  },

  async _apiFetch(url, options = {}) {
    const response = await fetch(url, options);
    if (response.status === 429) {
      const retryAfter = parseInt(response.headers.get("Retry-After") || "10", 10);
      this.startRateLimitCountdown(retryAfter);
      throw new Error(`HTTP 429 Rate Limit — Retry in ${retryAfter}s`);
    }
    return response;
  },

  startRateLimitCountdown(seconds) {
    this.rateLimitCountdown = seconds;
    const timer = setInterval(() => {
      this.rateLimitCountdown = Math.max(0, this.rateLimitCountdown - 1);
      if (this.rateLimitCountdown <= 0) clearInterval(timer);
    }, 1000);
  },

  async fetchMeta() {
    const response = await this._apiFetch("/api/meta");
    if (response.ok) {
      const data = await response.json();
      this.providers = data.providers || [];
      this.embeddingModels = data.embedding_models || [];
    }
  },

  async fetchSettings() {
    const response = await this._apiFetch("/api/settings");
    if (response.ok) {
      const data = await response.json();
      this.settings = { ...this.settings, ...data, api_key: "" };
      this.applyTheme();
      this._saveToLocalCache();
    }
  },

  onProviderChange() {
    const preset = this.providers.find(
      (provider) => provider.id === this.settings.provider
    );
    if (!preset) return;
    this.settings.base_url = preset.base_url;
    this.settings.model = preset.default_model;
    const savedMask =
      (this.settings.provider_keys_masked || {})[preset.id] || "";
    this.settings.api_key_masked = savedMask;
    this.settings.has_api_key = Boolean(savedMask);
    this.settings.api_key = "";
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
    const response = await this._apiFetch("/api/settings", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (response.ok) {
      const updated = await response.json();
      // `reindexed_chunks` is per-save metadata, not a persisted setting.
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
    this.setTheme(this.activeTheme.dark ? "brutal-light" : "brutal-dark");
  },

  _normalizeThemeId(themeId) {
    // Backwards compatibility with the original "dark" / "light" setting.
    const legacyThemeIds = { dark: "brutal-dark", light: "brutal-light" };
    const candidate = legacyThemeIds[themeId] || themeId;
    return this.themes.some((theme) => theme.id === candidate)
      ? candidate
      : "brutal-dark";
  },

  applyTheme(themeId = this.settings.theme) {
    const id = this._normalizeThemeId(themeId);
    const definition =
      this.themes.find((theme) => theme.id === id) || this.themes[0];

    this.settings.theme = definition.id;
    document.documentElement.setAttribute("data-theme", definition.id);
    document.documentElement.classList.toggle("dark", Boolean(definition.dark));
  },

  setTheme(themeId) {
    const id = this._normalizeThemeId(themeId);
    const changed = this.settings.theme !== id;
    this.applyTheme(id);
    if (changed) {
      this._saveToLocalCache();
      this.saveSettings();
    }
  },
};
