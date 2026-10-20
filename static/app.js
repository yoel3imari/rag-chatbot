/**
 * Alpine.js entry point.
 *
 * The component is assembled from focused modules: state/selectors, API client,
 * sessions, documents/ingestion, memory/export, evaluation, and chat streaming.
 */
(function () {
  const modules = window.RagModules || {};

  function mergeModules(target, sources) {
    for (const source of sources) {
      if (!source) continue;
      Object.defineProperties(target, Object.getOwnPropertyDescriptors(source));
    }
    return target;
  }

  window.ragApp = function ragApp() {
    return mergeModules({}, [
      modules.state,
      modules.api,
      modules.sessions,
      modules.documents,
      modules.memory,
      modules.evaluation,
      modules.chat,
    ]);
  };
})();
