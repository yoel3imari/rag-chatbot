window.RagModules = window.RagModules || {};

window.RagModules.memory = {
  async fetchMemories() {
    const response = await this._apiFetch("/api/memories");
    if (response.ok) {
      this.memories = await response.json();
      this.recalculateTokens();
    }
  },

  async addManualMemory() {
    if (!this.newMemoryFact.trim()) return;
    const response = await this._apiFetch("/api/memories", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        fact: this.newMemoryFact.trim(),
        category: "manual",
      }),
    });
    if (response.ok) {
      this.newMemoryFact = "";
      await this.fetchMemories();
    }
  },

  handleMemDeleteClick(memoryId) {
    if (this.armedMemDelete === memoryId) {
      clearTimeout(this._armTimers[`mem_${memoryId}`]);
      this.armedMemDelete = null;
      this.executeDeleteMemory(memoryId);
      return;
    }
    this.armedMemDelete = memoryId;
    clearTimeout(this._armTimers[`mem_${memoryId}`]);
    this._armTimers[`mem_${memoryId}`] = setTimeout(() => {
      if (this.armedMemDelete === memoryId) {
        this.armedMemDelete = null;
      }
    }, 3000);
  },

  disarmMemDelete(memoryId) {
    if (this.armedMemDelete === memoryId) {
      clearTimeout(this._armTimers[`mem_${memoryId}`]);
      this.armedMemDelete = null;
    }
  },

  async executeDeleteMemory(memoryId) {
    const response = await this._apiFetch(`/api/memories/${memoryId}`, {
      method: "DELETE",
    });
    if (response.ok) {
      this.memories = this.memories.filter((memory) => memory.id !== memoryId);
      this.recalculateTokens();
    }
  },

  async exportHistoryJson() {
    const response = await this._apiFetch("/api/export");
    if (!response.ok) return;
    const data = await response.json();
    this._downloadJson(
      data,
      `rag-chat-history-${new Date().toISOString().slice(0, 10)}.json`
    );
  },

  _downloadJson(data, filename) {
    const blob = new Blob([JSON.stringify(data, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    anchor.click();
    URL.revokeObjectURL(url);
  },
};
