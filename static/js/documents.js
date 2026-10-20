window.RagModules = window.RagModules || {};

window.RagModules.documents = {
  async fetchDocuments() {
    const query = this.currentSessionId
      ? `?session_id=${encodeURIComponent(this.currentSessionId)}`
      : "";
    const response = await this._apiFetch(`/api/documents${query}`);
    if (response.ok) {
      this.documents = await response.json();
    }
  },

  formatBytes(bytes) {
    const value = Number(bytes) || 0;
    if (value < 1024) return `${value} B`;
    if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
    return `${(value / (1024 * 1024)).toFixed(2)} MB`;
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
      // Scoped attachments: files dropped into the prompt use scope: session_id.
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
    const index = this.ingestToasts.findIndex((toast) => toast.id === toastData.id);
    if (index >= 0) {
      this.ingestToasts.splice(index, 1, {
        ...this.ingestToasts[index],
        ...toastData,
      });
    } else {
      this.ingestToasts.unshift(toastData);
    }
    if (toastData.status === "INDEXED" || toastData.status === "SYNCED") {
      setTimeout(() => {
        this.ingestToasts = this.ingestToasts.filter(
          (toast) => toast.id !== toastData.id
        );
      }, 4200);
    }
  },

  dismissToast(id) {
    this.ingestToasts = this.ingestToasts.filter((toast) => toast.id !== id);
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
      const response = await this._apiFetch("/api/ingest/file", {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        const errorJson = await response.json().catch(() => ({}));
        throw new Error(errorJson.detail || `Upload failed (${response.status})`);
      }
      await this._consumeIngestSSEStream(response, tempId);
    } catch (error) {
      this._upsertToast({
        id: tempId,
        name: file.name,
        status: "FAILED",
        progress: 100,
        message: error.message || "Upload failed",
      });
    }
  },

  async submitUrlWithSSE(urlString, scope = "global") {
    if (!urlString || !urlString.trim()) return;
    const cleanUrl = urlString.trim();
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
      const response = await this._apiFetch("/api/ingest/url", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          url: cleanUrl,
          scope,
          session_id: this.currentSessionId || "",
          idempotency_key: idempotencyKey,
        }),
      });
      if (!response.ok) {
        const errorJson = await response.json().catch(() => ({}));
        throw new Error(errorJson.detail || `URL ingest failed (${response.status})`);
      }
      await this._consumeIngestSSEStream(response, tempId);
    } catch (error) {
      this._upsertToast({
        id: tempId,
        name: cleanUrl,
        status: "FAILED",
        progress: 100,
        message: error.message || "URL ingestion failed",
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

  async resyncUrlDoc(documentRecord) {
    const tempId = documentRecord.id;
    this._upsertToast({
      id: tempId,
      name: documentRecord.name,
      source_type: "url",
      scope: documentRecord.scope,
      status: "QUEUED",
      progress: 10,
      message: `Re-syncing ${documentRecord.name}...`,
    });
    try {
      const response = await this._apiFetch(
        `/api/documents/${documentRecord.id}/resync`,
        { method: "POST" }
      );
      if (!response.ok) throw new Error(`Re-sync failed (${response.status})`);
      await this._consumeIngestSSEStream(response, tempId);
    } catch (error) {
      this._upsertToast({
        id: tempId,
        name: documentRecord.name,
        status: "FAILED",
        progress: 100,
        message: error.message || "Re-sync failed",
      });
    }
  },

  async _consumeIngestSSEStream(response, toastId) {
    const reader = response.body.getReader();
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
        const jsonString = line.slice(5).trim();
        try {
          const event = JSON.parse(jsonString);
          this._upsertToast({
            id: toastId,
            name: event.name,
            source_type: event.source_type,
            scope: event.scope,
            status: event.status,
            progress: event.progress || 50,
            message: event.message,
            error: event.error,
          });
          if (event.document && event.document.id) {
            const existingIndex = this.documents.findIndex(
              (documentRecord) => documentRecord.id === event.document.id
            );
            if (existingIndex >= 0) {
              this.documents.splice(existingIndex, 1, event.document);
            } else {
              this.documents.unshift(event.document);
            }
          }
        } catch (_) {}
      }
    }
    // Refresh once at end-of-stream rather than hammering /api/documents.
    await this.fetchDocuments();
  },

  // Two-step document delete.
  handleDocDeleteClick(docId) {
    if (this.armedDocDelete === docId) {
      clearTimeout(this._armTimers[`doc_${docId}`]);
      this.armedDocDelete = null;
      this.executeDeleteDocument(docId);
      return;
    }
    this.armedDocDelete = docId;
    clearTimeout(this._armTimers[`doc_${docId}`]);
    this._armTimers[`doc_${docId}`] = setTimeout(() => {
      if (this.armedDocDelete === docId) {
        this.armedDocDelete = null;
      }
    }, 3000);
  },

  disarmDocDelete(docId) {
    if (this.armedDocDelete === docId) {
      clearTimeout(this._armTimers[`doc_${docId}`]);
      this.armedDocDelete = null;
    }
  },

  async executeDeleteDocument(docId) {
    const documentRecord = this.documents.find(
      (documentItem) => documentItem.id === docId
    );
    const response = await this._apiFetch(`/api/documents/${docId}`, {
      method: "DELETE",
    });
    if (response.ok) {
      this.documents = this.documents.filter(
        (documentItem) => documentItem.id !== docId
      );
      if (documentRecord) {
        this.targetDocs = this.targetDocs.filter(
          (target) => target !== documentRecord.name
        );
      }
    }
  },
};
