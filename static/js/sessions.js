window.RagModules = window.RagModules || {};

window.RagModules.sessions = {
  async fetchSessions() {
    const response = await this._apiFetch("/api/sessions");
    if (response.ok) {
      this.sessions = await response.json();
      this._saveToLocalCache();
    }
  },

  async createNewSession() {
    if (this.isStreaming) this.stopGeneration();
    const response = await this._apiFetch("/api/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: "New Session" }),
    });
    if (response.ok) {
      const created = await response.json();
      this.sessions.unshift(created);
      await this.selectSession(created.id);
    }
  },

  async selectSession(sessionId) {
    if (this.isStreaming) this.stopGeneration();
    this.currentSessionId = sessionId;
    this.virtualLimit = 80;
    const response = await this._apiFetch(`/api/sessions/${sessionId}`);
    if (response.ok) {
      const data = await response.json();
      this.currentSession = data;
      this.messages = (data.messages || []).map((message) => ({
        ...message,
        showTrace: false,
      }));
      this.summaryDraft = data.summary || "";
      await this.fetchDocuments();
      this.recalculateTokens();
      this.scrollToBottom();
    }
  },

  // Two-step delete for sessions.
  handleSessionDeleteClick(sessionId) {
    if (this.armedSessionDelete === sessionId) {
      clearTimeout(this._armTimers[`sess_${sessionId}`]);
      this.armedSessionDelete = null;
      this.executeDeleteSession(sessionId);
      return;
    }
    this.armedSessionDelete = sessionId;
    clearTimeout(this._armTimers[`sess_${sessionId}`]);
    this._armTimers[`sess_${sessionId}`] = setTimeout(() => {
      if (this.armedSessionDelete === sessionId) {
        this.armedSessionDelete = null;
      }
    }, 3000);
  },

  disarmSessionDelete(sessionId) {
    if (this.armedSessionDelete === sessionId) {
      clearTimeout(this._armTimers[`sess_${sessionId}`]);
      this.armedSessionDelete = null;
    }
  },

  async executeDeleteSession(sessionId) {
    const response = await this._apiFetch(`/api/sessions/${sessionId}`, {
      method: "DELETE",
    });
    if (!response.ok) return;
    this.sessions = this.sessions.filter((session) => session.id !== sessionId);
    if (this.currentSessionId === sessionId) {
      if (this.sessions.length > 0) {
        await this.selectSession(this.sessions[0].id);
      } else {
        await this.createNewSession();
      }
    }
    await this.fetchDocuments();
  },

  // Two-step clear all sessions.
  handleClearAllSessionsClick() {
    if (this.armedClearAll) {
      this.armedClearAll = false;
      this.executeClearAllSessions();
      return;
    }
    this.armedClearAll = true;
    setTimeout(() => {
      this.armedClearAll = false;
    }, 3000);
  },

  async executeClearAllSessions() {
    const response = await this._apiFetch("/api/sessions", { method: "DELETE" });
    if (response.ok) {
      const data = await response.json();
      await this.fetchSessions();
      if (data.active_session) {
        await this.selectSession(data.active_session.id);
      }
      await this.fetchDocuments();
    }
  },

  async saveEditedSummary() {
    if (!this.currentSessionId) return;
    const response = await this._apiFetch(`/api/sessions/${this.currentSessionId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ summary: this.summaryDraft }),
    });
    if (response.ok) {
      const updated = await response.json();
      this.currentSession.summary = updated.summary;
      this.editingSummary = false;
      this.recalculateTokens();
    }
  },

  async toggleMessageContext(message) {
    const nextState = !message.included_in_context;
    message.included_in_context = nextState;
    this.recalculateTokens();
    const response = await this._apiFetch(`/api/messages/${message.id}/context`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ included_in_context: nextState }),
    });
    if (response.ok) {
      const updated = await response.json();
      message.included_in_context = updated.included_in_context;
      this.recalculateTokens();
    }
  },

  async linkOtherSession(targetSessionId) {
    if (
      !this.currentSessionId ||
      !targetSessionId ||
      targetSessionId === this.currentSessionId
    ) {
      return;
    }
    const response = await this._apiFetch(
      `/api/sessions/${this.currentSessionId}/link`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ target_session_id: targetSessionId }),
      }
    );
    if (response.ok) {
      const updated = await response.json();
      this.currentSession.linked_sessions = updated.linked_sessions || [];
    }
  },

  getSessionTitleById(sessionId) {
    const session = this.sessions.find((item) => item.id === sessionId);
    return session ? session.title : sessionId.slice(0, 8);
  },
};
