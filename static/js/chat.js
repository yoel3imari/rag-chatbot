window.RagModules = window.RagModules || {};

window.RagModules.chat = {
  // --- @ Mention & /link-session Autocomplete ---
  onPromptInput(event) {
    const element = event.target;
    element.style.height = "auto";
    element.style.height = Math.min(200, Math.max(48, element.scrollHeight)) + "px";

    this.recalculateTokens();
    this._checkAutocompleteTrigger(element);
  },

  _checkAutocompleteTrigger(textarea) {
    const cursor = textarea.selectionStart || 0;
    const textBefore = this.promptText.slice(0, cursor);

    const atMatch = textBefore.match(/(?:^|\s)@([^\s]*)$/);
    const slashMatch = textBefore.match(/(?:^|\s)\/link-session(?:\s+([^\s]*))?$/);

    if (slashMatch) {
      const query = (slashMatch[1] || "").toLowerCase();
      this.mentionTriggerStart = textBefore.lastIndexOf("/link-session");
      this.mentionQuery = query;
      this.mentionItems = this.sessions
        .filter(
          (session) =>
            session.id !== this.currentSessionId &&
            session.title.toLowerCase().includes(query)
        )
        .map((session) => ({
          kind: "session",
          id: session.id,
          label: session.title,
          badge: "SESSION",
        }));
      this.mentionOpen = this.mentionItems.length > 0;
      this.mentionSelectedIndex = 0;
      return;
    }

    if (atMatch) {
      const query = (atMatch[1] || "").toLowerCase();
      this.mentionTriggerStart = textBefore.lastIndexOf("@");
      this.mentionQuery = query;

      const documentItems = this.documents
        .filter(
          (documentRecord) =>
            (documentRecord.status === "INDEXED" ||
              documentRecord.status === "SYNCED") &&
            documentRecord.name.toLowerCase().includes(query)
        )
        .map((documentRecord) => ({
          kind: "doc",
          id: documentRecord.id,
          label: documentRecord.name,
          badge:
            documentRecord.scope === "global"
              ? `GLOBAL ${documentRecord.source_type.toUpperCase()}`
              : `SESSION ${documentRecord.source_type.toUpperCase()}`,
        }));

      const sessionItems = this.sessions
        .filter(
          (session) =>
            session.id !== this.currentSessionId &&
            (`session:${session.title}`.toLowerCase().includes(query) ||
              session.title.toLowerCase().includes(query))
        )
        .slice(0, 4)
        .map((session) => ({
          kind: "session",
          id: session.id,
          label: session.title,
          badge: "@SESSION",
        }));

      this.mentionItems = [...documentItems, ...sessionItems];
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
    this.targetDocs = this.targetDocs.filter((target) => target !== docName);
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

    // Parse inline @docName mentions typed directly in promptText.
    const activeDocNames = [...this.targetDocs];
    for (const documentRecord of this.documents) {
      if (
        text.includes(`@${documentRecord.name}`) &&
        !activeDocNames.includes(documentRecord.name)
      ) {
        activeDocNames.push(documentRecord.name);
      }
    }

    this.promptText = "";
    this.mentionOpen = false;
    const textarea = document.getElementById("chat-prompt-textarea");
    if (textarea) textarea.style.height = "48px";

    const temporaryUserMessage = {
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

    this.messages.push(temporaryUserMessage, assistantPlaceholder);
    const assistantIndex = this.messages.length - 1;
    const userIndex = this.messages.length - 2;

    this.isStreaming = true;
    this.abortController = new AbortController();
    this.scrollToBottom();

    let clientHeaderKey = "";
    try {
      const encodedKey = sessionStorage.getItem(
        `rag_client_api_key_${this.settings.provider}`
      );
      if (encodedKey) clientHeaderKey = atob(encodedKey);
    } catch (_) {}

    const headers = { "Content-Type": "application/json" };
    if (clientHeaderKey) {
      headers["X-LLM-API-Key"] = clientHeaderKey;
    }

    try {
      const response = await this._apiFetch("/api/chat/stream", {
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

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Chat error (${response.status})`);
      }

      const reader = response.body.getReader();
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
            const event = JSON.parse(rawJson);
            if (event.type === "session_titled") {
              const session = this.sessions.find(
                (item) => item.id === event.session_id
              );
              if (session) session.title = event.title;
              if (this.currentSession) this.currentSession.title = event.title;
            } else if (event.type === "user_message_saved" && event.message) {
              this.messages[userIndex] = {
                ...event.message,
                showTrace: false,
              };
            } else if (event.type === "react_step") {
              this.messages[assistantIndex].react_steps.push(event);
              this.scrollToBottom();
            } else if (event.type === "citations") {
              this.messages[assistantIndex].citations = event.citations || [];
              this.scrollToBottom();
            } else if (event.type === "token") {
              this.messages[assistantIndex].content += event.delta || "";
              this.scrollToBottom();
            } else if (event.type === "summary_updated") {
              if (this.currentSession) {
                this.currentSession.summary = event.summary;
                this.summaryDraft = event.summary;
              }
            } else if (event.type === "memory_extracted" && event.memory) {
              await this.fetchMemories();
            } else if (event.type === "done" && event.message) {
              this.messages[assistantIndex] = {
                ...event.message,
                showTrace: false,
                isStreaming: false,
              };
            }
          } catch (_) {}
        }
      }
    } catch (error) {
      if (error.name === "AbortError") {
        this.messages[assistantIndex].content +=
          "\n\n`[GENERATION ABORTED BY USER]`";
      } else if (this.messages[assistantIndex].content) {
        this.messages[assistantIndex].content += `\n\n⚠️ **Error:** ${error.message}`;
      } else {
        this.messages[assistantIndex].content = `⚠️ **Error:** ${error.message}`;
      }
      this.messages[assistantIndex].isStreaming = false;
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

    html = html.replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g, (_, language, code) => {
      const badge = language
        ? `<div class="text-xs font-mono uppercase tracking-wider px-3.5 py-1.5 border-b border-[var(--border)] text-[var(--text-muted)]">${language}</div>`
        : "";
      return `<div class="my-3.5 border border-[var(--border-strong)] bg-[var(--bg-elevated)] overflow-x-auto">${badge}<pre class="p-3.5 text-sm font-mono leading-relaxed"><code>${code.trim()}</code></pre></div>`;
    });

    html = html.replace(
      /`([^`]+)`/g,
      '<code class="px-1.5 py-0.5 text-sm font-mono border border-[var(--border-strong)] bg-[var(--bg-hover)]">$1</code>'
    );

    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong class="font-semibold">$1</strong>');
    html = html.replace(
      /(^|[\s(])\*([^\s*][^*]*?)\*(?=[\s).,!?:;]|$)/g,
      '$1<em class="italic text-[var(--text-secondary)]">$2</em>'
    );

    html = html.replace(
      /^&gt;\s?(.*)$/gm,
      '<blockquote class="border-l-2 border-[var(--accent)] pl-3.5 py-0.5 my-2 text-[var(--text-muted)]">$1</blockquote>'
    );

    html = html.replace(
      /\[([a-zA-Z0-9_.\-/]+\.(?:pdf|txt|md|html)|https?:\/\/[^\]]+)\]/gi,
      '<span class="inline-block px-1.5 py-0.5 text-xs font-mono border border-[var(--accent)] bg-[var(--accent)] text-[var(--accent-contrast)] font-medium">[$1]</span>'
    );

    html = html.replace(/\n\n/g, "</p><p class='mt-3'>");
    html = html.replace(/\n/g, "<br>");
    return `<p>${html}</p>`;
  },
};
