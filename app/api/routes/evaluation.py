"""RAG and ReAct evaluation routes."""

import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.evaluation import (
    GOLDEN_EVAL_DATASET,
    generate_synthetic_eval_cases,
    run_evaluation,
    run_evaluation_stream,
)
from app.models import EvalRunRequest, SyntheticEvalRequest
from app.store import store

router = APIRouter(tags=["evaluation"])

SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


@router.get("/api/eval/dataset")
async def get_default_eval_dataset():
    return {
        "name": "Golden Benchmark Suite",
        "cases": GOLDEN_EVAL_DATASET,
    }


@router.post("/api/eval/generate")
async def generate_synthetic_dataset(payload: SyntheticEvalRequest):
    cases = await generate_synthetic_eval_cases(
        doc_ids=payload.doc_ids,
        max_cases=max(1, min(30, payload.max_cases)),
        use_llm=payload.use_llm,
    )
    return {
        "name": f"Synthetic Corpus Suite ({len(cases)} cases)",
        "cases": cases,
    }


@router.get("/api/eval/runs")
async def list_evaluation_runs():
    return store.list_eval_runs()


@router.get("/api/eval/runs/{run_id}")
async def get_evaluation_run(run_id: str):
    run = store.get_eval_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    return run


@router.delete("/api/eval/runs/{run_id}")
async def delete_evaluation_run(run_id: str):
    deleted = store.delete_eval_run(run_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    return {"deleted": True, "id": run_id}


@router.post("/api/eval/run")
async def trigger_evaluation_run(payload: EvalRunRequest):
    return await run_evaluation(
        mode=payload.mode,
        dataset_source=payload.dataset_source,
        custom_cases=payload.custom_cases,
        top_k=payload.top_k,
        similarity_threshold=payload.similarity_threshold,
        dedup_threshold=payload.dedup_threshold,
        embedding_model=payload.embedding_model,
        max_synthetic_cases=payload.max_synthetic_cases,
        persist=True,
    )


@router.post("/api/eval/run/stream")
async def trigger_evaluation_stream(payload: EvalRunRequest):
    async def _sse_eval_generator():
        async for event in run_evaluation_stream(
            mode=payload.mode,
            dataset_source=payload.dataset_source,
            custom_cases=payload.custom_cases,
            top_k=payload.top_k,
            similarity_threshold=payload.similarity_threshold,
            dedup_threshold=payload.dedup_threshold,
            embedding_model=payload.embedding_model,
            max_synthetic_cases=payload.max_synthetic_cases,
            persist=True,
        ):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        _sse_eval_generator(), media_type="text/event-stream", headers=SSE_HEADERS
    )
