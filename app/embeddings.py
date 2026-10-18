import threading
from typing import Any, Dict, List, Optional
import numpy as np
from langchain_core.embeddings import Embeddings
from crispembed import CrispEmbed

from app.config import EMBEDDING_DIM


MODEL_ALIASES = {
    "bge-small": "all-MiniLM-L6-v2",
    "bge-small-en-v1.5": "all-MiniLM-L6-v2",
    "bge-m3": "multilingual-e5-small",
}


class EmbeddingEngine(Embeddings):
    """Thread-safe CrispEmbed LangChain Embeddings implementation with cosine deduplication."""

    def __init__(self, default_model: str = "all-MiniLM-L6-v2"):
        self._lock = threading.RLock()
        self._current_model_name: str = ""
        self._model: Optional[CrispEmbed] = None
        self.load_model(default_model)

    def load_model(self, model_name: str) -> CrispEmbed:
        resolved = MODEL_ALIASES.get(model_name, model_name) or "all-MiniLM-L6-v2"
        with self._lock:
            if self._model is not None and self._current_model_name == resolved:
                return self._model
            try:
                candidate = CrispEmbed(resolved)
                if int(candidate.dim) != EMBEDDING_DIM:
                    raise ValueError(
                        f"Model '{resolved}' has dimension {candidate.dim}, expected {EMBEDDING_DIM}."
                    )
                self._model = candidate
                self._current_model_name = resolved
            except Exception:
                self._model = CrispEmbed("all-MiniLM-L6-v2")
                self._current_model_name = "all-MiniLM-L6-v2"
            assert int(self._model.dim) == EMBEDDING_DIM, (
                f"Embedding dimension mismatch: {self._model.dim} != {EMBEDDING_DIM}"
            )
            return self._model

    @property
    def dim(self) -> int:
        with self._lock:
            if self._model is None:
                self.load_model("all-MiniLM-L6-v2")
            return int(self._model.dim)

    @property
    def model_name(self) -> str:
        return self._current_model_name

    def embed_query(self, text: str, model_name: Optional[str] = None) -> List[float]:
        with self._lock:
            if model_name:
                self.load_model(model_name)
            vec = self._model.encode(text or " ", normalize=True)
            return np.asarray(vec, dtype=np.float32).flatten().tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_batch(texts)

    def embed_batch(self, texts: List[str], model_name: Optional[str] = None) -> List[List[float]]:
        if not texts:
            return []
        with self._lock:
            if model_name:
                self.load_model(model_name)
            cleaned = [t if t and t.strip() else " " for t in texts]
            arr = self._model.encode(cleaned, normalize=True)
            arr_np = np.asarray(arr, dtype=np.float32)
            if arr_np.ndim == 1:
                return [arr_np.tolist()]
            return [row.tolist() for row in arr_np]

    @staticmethod
    def deduplicate_chunks(
        chunks: List[Dict[str, Any]], threshold: float = 0.92
    ) -> List[Dict[str, Any]]:
        """Deduplicates overlapping chunks using cosine similarity thresholding before prompt injection."""
        if len(chunks) <= 1:
            return chunks
        kept: List[Dict[str, Any]] = []
        kept_vecs: List[np.ndarray] = []
        for item in chunks:
            vec = item.get("vector")
            if vec is not None and len(vec) > 0:
                v = np.asarray(vec, dtype=np.float32)
                norm = np.linalg.norm(v)
                if norm > 0:
                    v = v / norm
                is_dup = False
                for kv in kept_vecs:
                    if float(np.dot(v, kv)) >= threshold:
                        is_dup = True
                        break
                if not is_dup:
                    kept.append(item)
                    kept_vecs.append(v)
            else:
                words_a = set(item.get("text", "").lower().split())
                is_dup = False
                for k_item in kept:
                    words_b = set(k_item.get("text", "").lower().split())
                    if words_a and words_b:
                        jaccard = len(words_a & words_b) / float(len(words_a | words_b))
                        if jaccard >= threshold:
                            is_dup = True
                            break
                if not is_dup:
                    kept.append(item)
        return kept


embedding_engine = EmbeddingEngine("all-MiniLM-L6-v2")
