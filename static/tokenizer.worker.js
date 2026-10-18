// Web Worker for real-time client-side heuristic token pre-counting (aligned with app/store.py)
function estimateTextTokens(text) {
  if (!text) return 0;
  const trimmed = String(text).trim();
  if (!trimmed) return 0;
  const words = trimmed.split(/\s+/).length;
  const chars = trimmed.length;
  return Math.max(1, Math.round((chars / 3.8 + words * 1.3) / 2));
}

self.onmessage = function (e) {
  const {
    promptText = "",
    systemPrompt = "",
    sessionSummary = "",
    messages = [],
    memories = [],
    useMemory = true,
    maxContextTokens = 8192,
    reservedOutputTokens = 2048,
  } = e.data || {};

  const promptTokens = estimateTextTokens(promptText);
  const systemTokens = estimateTextTokens(systemPrompt);
  const summaryTokens = estimateTextTokens(sessionSummary);

  let memoryTokens = 0;
  if (useMemory && Array.isArray(memories)) {
    for (const m of memories) {
      memoryTokens += estimateTextTokens(m.fact || "");
    }
    memoryTokens = Math.min(memoryTokens, 400);
  }

  let historyTokens = 0;
  if (Array.isArray(messages)) {
    for (const msg of messages) {
      if (msg.included_in_context !== false) {
        historyTokens += msg.tokens ? Number(msg.tokens) : estimateTextTokens(msg.content || "");
      }
    }
  }

  const activeInputTokens =
    promptTokens + systemTokens + summaryTokens + memoryTokens + historyTokens;
  const hardInputLimit = Math.max(512, Number(maxContextTokens) - Number(reservedOutputTokens));
  // Block submission when the single prompt exceeds available input budget
  // (prior conversation history is automatically compacted/windowed by the backend).
  const exceedsHardLimit = promptTokens > hardInputLimit;

  self.postMessage({
    promptTokens,
    systemTokens,
    summaryTokens,
    memoryTokens,
    historyTokens,
    activeInputTokens,
    hardInputLimit,
    maxContextTokens: Number(maxContextTokens),
    reservedOutputTokens: Number(reservedOutputTokens),
    exceedsHardLimit,
  });
};
