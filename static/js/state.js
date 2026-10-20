window.RagModules = window.RagModules || {};

window.RagModules.state = {
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
  activeSettingsTab: "llm", // llm | docs | urls | general | memory | eval
  showApiKey: false,
  settingsSavedBanner: false,
  lastSavedNote: "",

  // --- Chat & Prompt Input ---
  promptText: "",
  isStreaming: false,
  abortController: null,
  useGlobalDocs: true,
  useMemory: true,
  targetDocs: [], // source names tagged via @
  showInlineUrlInput: false,
  inlineUrlValue: "",
  inlineIngestScope: "session", // "session" | "global"
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
  ingestToasts: [],
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
    return this.documents.filter((document) => document.source_type === "file");
  },

  get urlDocuments() {
    return this.documents.filter((document) => document.source_type === "url");
  },

  get currentProviderPreset() {
    return (
      this.providers.find((provider) => provider.id === this.settings.provider) || {
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
};
