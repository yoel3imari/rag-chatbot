window.RagModules = window.RagModules || {};

window.RagModules.evaluation = {
  openEvaluationTab() {
    this.settingsOpen = true;
    this.activeSettingsTab = "eval";
    this.fetchEvalRuns();
  },

  async fetchEvalRuns() {
    try {
      const response = await this._apiFetch("/api/eval/runs");
      if (response.ok) {
        this.evalRuns = await response.json();
        if (!this.activeEvalRun && this.evalRuns.length > 0) {
          this.activeEvalRun = this.evalRuns[0];
        }
      }
    } catch (_) {}
  },

  async selectEvalRunById(runId) {
    if (!runId) return;
    const found = this.evalRuns.find((run) => run.id === runId);
    if (found) {
      this.activeEvalRun = found;
      this.expandedEvalCases = {};
      return;
    }
    const response = await this._apiFetch(`/api/eval/runs/${runId}`);
    if (response.ok) {
      this.activeEvalRun = await response.json();
      this.expandedEvalCases = {};
    }
  },

  async deleteEvalRun(runId) {
    if (!runId) return;
    const response = await this._apiFetch(`/api/eval/runs/${runId}`, {
      method: "DELETE",
    });
    if (response.ok) {
      this.evalRuns = this.evalRuns.filter((run) => run.id !== runId);
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
    const response = await this._apiFetch("/api/eval/dataset");
    if (response.ok) {
      const data = await response.json();
      this.customEvalJsonText = JSON.stringify(data.cases || [], null, 2);
    }
  },

  async generateSyntheticPreview() {
    this.evalCustomError = "";
    this.evalStatusMessage =
      "Generating synthetic Q&A test cases from indexed LanceDB chunks...";
    try {
      const response = await this._apiFetch("/api/eval/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ max_cases: 6, use_llm: false }),
      });
      if (response.ok) {
        const data = await response.json();
        this.customEvalJsonText = JSON.stringify(data.cases || [], null, 2);
        this.evalConfig.dataset_source = "custom";
      }
    } catch (error) {
      this.evalCustomError = error.message || "Failed to generate synthetic cases";
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
          this.evalCustomError =
            "Custom JSON dataset must contain at least 1 test case.";
          return;
        }
      } catch (error) {
        this.evalCustomError = `Invalid JSON: ${error.message}`;
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
      const response = await this._apiFetch("/api/eval/run/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        const errorJson = await response.json().catch(() => ({}));
        throw new Error(errorJson.detail || `Evaluation failed (${response.status})`);
      }

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
          const rawJson = line.slice(5).trim();
          try {
            const event = JSON.parse(rawJson);
            if (event.type === "eval_start") {
              this.evalProgress = 10;
              this.evalStatusMessage = `Running ${event.dataset_name} (${event.total_cases} cases)...`;
            } else if (event.type === "case_result") {
              this.evalProgress = event.progress || 50;
              this.evalStatusMessage = `Evaluated case ${event.index}/${event.total_cases}: ${event.result?.case_id}`;
            } else if (event.type === "eval_complete" && event.run) {
              this.evalProgress = 100;
              this.activeEvalRun = event.run;
              await this.fetchEvalRuns();
            }
          } catch (_) {}
        }
      }
    } catch (error) {
      this.evalCustomError = error.message || "Evaluation run failed";
    } finally {
      this.isRunningEval = false;
      this.evalStatusMessage = "";
    }
  },

  exportActiveEvalJson() {
    if (!this.activeEvalRun) return;
    this._downloadJson(
      this.activeEvalRun,
      `rag-eval-${this.activeEvalRun.mode}-${new Date().toISOString().slice(0, 10)}.json`
    );
  },
};
