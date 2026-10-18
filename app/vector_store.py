import threading
import warnings
from typing import Any, Dict, List, Optional
import lancedb
import numpy as np
import pyarrow as pa

from app.config import EMBEDDING_DIM, LANCEDB_DIR
from app.embeddings import embedding_engine


def _escape_sql(val: str) -> str:
    return val.replace("'", "''")


class VectorStore:
    """Embedded LanceDB vector store with hybrid search (Vector + Tantivy FTS + RRF + Cosine Deduplication)."""

    TABLE_NAME = "rag_chunks"

    def __init__(self, uri: str = str(LANCEDB_DIR)):
        assert embedding_engine.dim == EMBEDDING_DIM, (
            f"CrispEmbed model dimension ({embedding_engine.dim}) does not match LanceDB schema ({EMBEDDING_DIM})"
        )
        self.uri = uri
        self._lock = threading.RLock()
        self._fts_dirty = False
        # Cache of the corpus-wide embedding model. None means "unknown/mixed — verify on demand".
        self._uniform_corpus_model: Optional[str] = None
        self.db = lancedb.connect(self.uri)
        self._ensure_table()

    def _schema(self, dim: int = EMBEDDING_DIM) -> pa.Schema:
        return pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("doc_id", pa.string()),
                pa.field("source_name", pa.string()),
                pa.field("source_type", pa.string()),
                pa.field("scope", pa.string()),
                pa.field("session_id", pa.string()),
                pa.field("user_id", pa.string()),
                pa.field("embedding_model", pa.string()),
                pa.field("chunk_index", pa.int32()),
                pa.field("text", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), dim)),
            ]
        )

    def _ensure_table(self) -> None:
        with self._lock:
            res = self.db.list_tables()
            existing = res.tables if hasattr(res, "tables") else list(res)
            if self.TABLE_NAME not in existing:
                self.table = self.db.create_table(
                    self.TABLE_NAME, schema=self._schema(EMBEDDING_DIM), mode="create"
                )
            else:
                self.table = self.db.open_table(self.TABLE_NAME)
                # Ensure schema includes embedding_model column (upgrade older tables cleanly)
                field_names = set(self.table.schema.names)
                if "embedding_model" not in field_names:
                    rows = self.table.to_arrow().to_pylist()
                    for r in rows:
                        r["embedding_model"] = embedding_engine.model_name
                    self.table = self.db.create_table(
                        self.TABLE_NAME, schema=self._schema(EMBEDDING_DIM), mode="overwrite"
                    )
                    if rows:
                        self.table.add(rows)
                        self._fts_dirty = True
                    self._uniform_corpus_model = embedding_engine.model_name

    def _ensure_fts_fresh_unlocked(self) -> None:
        """Lazily rebuilds the FTS index only when dirty and before an FTS query executes."""
        if not self._fts_dirty:
            return
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.table.create_fts_index("text", replace=True)
            self._fts_dirty = False
        except Exception:
            pass

    def add_chunks(self, records: List[Dict[str, Any]]) -> int:
        if not records:
            return 0
        active_model = embedding_engine.model_name
        for r in records:
            if "embedding_model" not in r or not r["embedding_model"]:
                r["embedding_model"] = active_model
        with self._lock:
            self.table.add(records)
            self._fts_dirty = True
            if self._uniform_corpus_model != active_model:
                # New rows carry the active model; if the cache disagrees, the
                # corpus may now be mixed — invalidate so the next check scans.
                self._uniform_corpus_model = None
            return len(records)

    def delete_by_doc_ids(self, doc_ids: List[str]) -> None:
        if not doc_ids:
            return
        with self._lock:
            quoted = ", ".join(f"'{_escape_sql(d)}'" for d in doc_ids if d)
            if not quoted:
                return
            try:
                self.table.delete(f"doc_id IN ({quoted})")
                self._fts_dirty = True
            except Exception:
                pass

    def reindex_for_model(self, target_model: str) -> int:
        """
        Re-embeds only the stored chunks whose `embedding_model` differs from `target_model`,
        preventing mixed-model vector spaces after an embedding model switch.

        Mismatch detection uses a projected column scan (id + embedding_model only), so the
        steady state skips materializing text + vectors of the whole corpus. Safe to call on
        every retrieval.
        """
        with self._lock:
            embedding_engine.load_model(target_model)
            resolved_model = embedding_engine.model_name

            # Fast path: corpus is known to be uniform at the requested model.
            if self._uniform_corpus_model == resolved_model:
                return 0

            try:
                total_rows = self.table.count_rows()
            except Exception:
                return 0
            if total_rows == 0:
                self._uniform_corpus_model = resolved_model
                return 0

            all_rows: Optional[List[Dict[str, Any]]] = None
            mismatch_ids: List[str] = []
            try:
                # Column projection — avoids loading vector/text payloads for the check.
                proj = (
                    self.table.search()
                    .select(["id", "embedding_model"])
                    .limit(total_rows)
                    .to_arrow()
                )
                ids = proj.column("id").to_pylist()
                models = proj.column("embedding_model").to_pylist()
                mismatch_ids = [i for i, m in zip(ids, models) if m != resolved_model]
            except Exception:
                # Fallback: full scan for the check.
                try:
                    all_rows = self.table.to_arrow().to_pylist()
                except Exception:
                    return 0
                mismatch_ids = [
                    r["id"] for r in all_rows if r.get("embedding_model") != resolved_model
                ]

            if not mismatch_ids:
                self._uniform_corpus_model = resolved_model
                return 0

            if all_rows is None:
                try:
                    all_rows = self.table.to_arrow().to_pylist()
                except Exception:
                    return 0
            lookup = {r["id"]: r for r in all_rows}
            mismatch_rows = [lookup[i] for i in mismatch_ids if i in lookup]
            if not mismatch_rows:
                self._uniform_corpus_model = resolved_model
                return 0

            try:
                texts = [r.get("text", " ") for r in mismatch_rows]
                new_vectors = embedding_engine.embed_batch(texts, model_name=resolved_model)
                if not new_vectors or len(new_vectors) != len(mismatch_rows):
                    # Partial/failed embedding batch — leave the corpus untouched
                    # rather than writing partially-initialized vectors.
                    return 0
            except Exception:
                return 0

            vec_by_id = {r["id"]: vec for r, vec in zip(mismatch_rows, new_vectors)}
            for row in all_rows:
                vec = vec_by_id.get(row["id"])
                if vec is not None:
                    row["vector"] = vec
                    row["embedding_model"] = resolved_model

            self.table = self.db.create_table(
                self.TABLE_NAME, schema=self._schema(EMBEDDING_DIM), mode="overwrite"
            )
            self.table.add(all_rows)
            self._fts_dirty = True
            self._uniform_corpus_model = resolved_model
            return len(vec_by_id)

    def _build_where_clause(
        self,
        user_id: str = "default",
        session_id: Optional[str] = None,
        use_global_docs: bool = True,
        target_docs: Optional[List[str]] = None,
    ) -> str:
        clauses = [f"user_id = '{_escape_sql(user_id)}'"]

        scope_parts = []
        if use_global_docs:
            scope_parts.append("scope = 'global'")
        if session_id:
            s_esc = _escape_sql(session_id)
            scope_parts.append(f"scope = '{s_esc}'")
            scope_parts.append(f"session_id = '{s_esc}'")

        if scope_parts:
            clauses.append("(" + " OR ".join(scope_parts) + ")")
        else:
            if not target_docs:
                clauses.append("scope = '__none__'")

        if target_docs:
            target_conds = []
            for t in target_docs:
                t_esc = _escape_sql(t.strip())
                if t_esc:
                    target_conds.append(f"doc_id = '{t_esc}'")
                    target_conds.append(f"source_name = '{t_esc}'")
            if target_conds:
                clauses.append("(" + " OR ".join(target_conds) + ")")

        return " AND ".join(clauses)

    def hybrid_search(
        self,
        query: str,
        user_id: str = "default",
        session_id: Optional[str] = None,
        use_global_docs: bool = True,
        target_docs: Optional[List[str]] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.15,
        dedup_threshold: float = 0.92,
        embedding_model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Executes hybrid search (Dense Vector + Tantivy BM25 FTS) with RRF fusion and cosine deduplication."""
        if not query or not query.strip():
            return []

        with self._lock:
            try:
                total_rows = self.table.count_rows()
                if total_rows == 0:
                    return []
            except Exception:
                return []

            if embedding_model:
                self.reindex_for_model(embedding_model)

            self._ensure_fts_fresh_unlocked()

            where_sql = self._build_where_clause(
                user_id=user_id,
                session_id=session_id,
                use_global_docs=use_global_docs,
                target_docs=target_docs,
            )

            q_vec = embedding_engine.embed_query(query, model_name=embedding_model)
            q_np = np.asarray(q_vec, dtype=np.float32)
            q_norm = np.linalg.norm(q_np)
            if q_norm > 0:
                q_np = q_np / q_norm

            fetch_k = max(top_k * 4, 20)

            # 1. Dense Vector Search
            vec_results: List[Dict[str, Any]] = []
            try:
                vec_results = (
                    self.table.search(q_vec)
                    .where(where_sql, prefilter=True)
                    .limit(fetch_k)
                    .to_list()
                )
            except Exception:
                try:
                    vec_results = self.table.search(q_vec).where(where_sql).limit(fetch_k).to_list()
                except Exception:
                    vec_results = []

            # 2. Full-Text Search (BM25 via Tantivy)
            fts_results: List[Dict[str, Any]] = []
            try:
                fts_results = (
                    self.table.search(query, query_type="fts")
                    .where(where_sql, prefilter=True)
                    .limit(fetch_k)
                    .to_list()
                )
            except Exception:
                try:
                    fts_results = (
                        self.table.search(query, query_type="fts")
                        .where(where_sql)
                        .limit(fetch_k)
                        .to_list()
                    )
                except Exception:
                    fts_results = []

            # 3. Reciprocal Rank Fusion (RRF) + Exact Cosine Similarity Computation
            rrf_k = 60.0
            merged: Dict[str, Dict[str, Any]] = {}

            for rank, row in enumerate(vec_results):
                cid = row["id"]
                c_vec = np.asarray(row.get("vector", []), dtype=np.float32)
                c_norm = np.linalg.norm(c_vec)
                cos_sim = float(np.dot(q_np, c_vec / c_norm)) if c_norm > 0 and len(c_vec) == len(q_np) else 0.0
                merged[cid] = {
                    **row,
                    "cosine_similarity": cos_sim,
                    "vec_rank": rank + 1,
                    "fts_rank": None,
                    "rrf_score": 1.0 / (rrf_k + rank + 1),
                    "match_type": "vector",
                }

            for rank, row in enumerate(fts_results):
                cid = row["id"]
                fts_contrib = 1.0 / (rrf_k + rank + 1)
                if cid in merged:
                    merged[cid]["fts_rank"] = rank + 1
                    merged[cid]["rrf_score"] += fts_contrib
                    merged[cid]["match_type"] = "hybrid (vector+fts)"
                else:
                    c_vec = np.asarray(row.get("vector", []), dtype=np.float32)
                    c_norm = np.linalg.norm(c_vec)
                    cos_sim = float(np.dot(q_np, c_vec / c_norm)) if c_norm > 0 and len(c_vec) == len(q_np) else 0.0
                    merged[cid] = {
                        **row,
                        "cosine_similarity": cos_sim,
                        "vec_rank": None,
                        "fts_rank": rank + 1,
                        "rrf_score": fts_contrib,
                        "match_type": "fts",
                    }

            candidates = []
            for item in merged.values():
                cos = item["cosine_similarity"]
                has_fts = item["fts_rank"] is not None and item["fts_rank"] <= 5
                if cos >= similarity_threshold or has_fts:
                    item["hybrid_score"] = round(0.65 * max(0.0, cos) + 0.35 * (item["rrf_score"] * 30.0), 4)
                    candidates.append(item)

            candidates.sort(key=lambda x: (x["hybrid_score"], x["cosine_similarity"]), reverse=True)

            # 4. Deduplicate overlapping chunks via cosine similarity thresholding
            deduped = embedding_engine.deduplicate_chunks(candidates, threshold=dedup_threshold)
            top_items = deduped[:top_k]

            clean_results = []
            for r in top_items:
                clean_results.append(
                    {
                        "id": r["id"],
                        "doc_id": r["doc_id"],
                        "source_name": r["source_name"],
                        "source_type": r["source_type"],
                        "scope": r["scope"],
                        "embedding_model": r.get("embedding_model", embedding_engine.model_name),
                        "chunk_index": r["chunk_index"],
                        "text": r["text"],
                        "similarity": round(r["cosine_similarity"], 4),
                        "hybrid_score": r["hybrid_score"],
                        "match_type": r["match_type"],
                    }
                )
            return clean_results


vector_store = VectorStore()
