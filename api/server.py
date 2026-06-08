"""
Orion API — Phase 1
FastAPI wrapper around the agent core.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal
import json
import os

from agent.core import OrionAgent

app = FastAPI(
    title="Orion Agent API",
    description="AI-powered job search pipeline agent",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory trace store — Phase 3 will replace with vector DB
_trace_store: dict = {}

agent = OrionAgent()


# ── Request / Response models ────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    job_description: str = Field(..., min_length=50, description="Full job posting text")
    resume_text: str = Field(..., min_length=50, description="Resume as plain text")
    candidate_name: str = Field(default="Candidate")
    tone: Literal["professional", "conversational", "enthusiastic"] = "professional"


class TraceResponse(BaseModel):
    run_id: str
    status: str
    goal: str
    total_duration_ms: float
    steps: list
    final_output: dict


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}


@app.post("/analyze", response_model=TraceResponse)
async def analyze(req: AnalyzeRequest):
    """
    Run the agent on a job posting + resume.
    Returns a full execution trace + structured output.
    """
    try:
        trace = await agent.run(
            job_description=req.job_description,
            resume_text=req.resume_text,
            candidate_name=req.candidate_name,
            tone=req.tone,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    trace_dict = trace.to_dict()
    _trace_store[trace.run_id] = trace_dict
    return trace_dict


@app.get("/traces")
async def list_traces():
    """List all stored agent run traces (most recent first)."""
    traces = sorted(
        _trace_store.values(),
        key=lambda t: t["started_at"],
        reverse=True,
    )
    # Return lightweight summaries
    return [
        {
            "run_id": t["run_id"],
            "started_at": t["started_at"],
            "status": t["status"],
            "total_duration_ms": t["total_duration_ms"],
            "fit_score": t.get("final_output", {}).get("fit_score"),
            "goal": t["goal"],
        }
        for t in traces
    ]


@app.get("/traces/{run_id}", response_model=TraceResponse)
async def get_trace(run_id: str):
    """Get full trace for a specific run."""
    trace = _trace_store.get(run_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")
    return trace
