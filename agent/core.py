"""
Orion Agent Core — Phase 1
The main agent loop: plan → select tool → execute → store trace
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any

import anthropic

from agent.tools.fit_scorer import score_fit
from agent.tools.email_drafter import draft_email
from agent.tools.resume_parser import parse_resume

# ── Tool Registry ────────────────────────────────────────────────────────────
# Each tool the agent can call is registered here.
# Phase 2 will make this dynamic; for now it's explicit.

TOOLS = [
    {
        "name": "score_fit",
        "description": (
            "Scores how well a job posting matches a candidate resume. "
            "Returns a 0-100 fit score, a list of matched skills, a list of "
            "skill gaps, and a short summary."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "job_description": {"type": "string", "description": "Full job posting text"},
                "resume_text": {"type": "string", "description": "Candidate resume as plain text"},
            },
            "required": ["job_description", "resume_text"],
        },
    },
    {
        "name": "draft_email",
        "description": (
            "Drafts a short, personalized cold outreach email to a recruiter or "
            "hiring manager based on the job posting and fit analysis."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "job_description": {"type": "string"},
                "candidate_name": {"type": "string"},
                "fit_summary": {"type": "string", "description": "Output from score_fit"},
                "tone": {
                    "type": "string",
                    "enum": ["professional", "conversational", "enthusiastic"],
                    "description": "Tone of the email",
                },
            },
            "required": ["job_description", "candidate_name", "fit_summary"],
        },
    },
    {
        "name": "parse_resume",
        "description": "Extracts structured information from raw resume text: skills, experience, education.",
        "input_schema": {
            "type": "object",
            "properties": {
                "resume_text": {"type": "string"},
            },
            "required": ["resume_text"],
        },
    },
]

TOOL_FN_MAP = {
    "score_fit": score_fit,
    "draft_email": draft_email,
    "parse_resume": parse_resume,
}


# ── Trace & Step dataclasses ─────────────────────────────────────────────────

@dataclass
class AgentStep:
    step_index: int
    tool_name: str
    tool_input: dict
    tool_output: Any
    duration_ms: float
    status: str = "success"  # success | error


@dataclass
class AgentTrace:
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    goal: str = ""
    steps: list[AgentStep] = field(default_factory=list)
    final_output: dict = field(default_factory=dict)
    status: str = "running"  # running | complete | failed
    total_duration_ms: float = 0.0

    def to_dict(self):
        d = asdict(self)
        return d


# ── Agent ────────────────────────────────────────────────────────────────────

class OrionAgent:
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.Anthropic()
        self.model = model

    async def run(
        self,
        job_description: str,
        resume_text: str,
        candidate_name: str = "Candidate",
        tone: str = "professional",
    ) -> AgentTrace:
        trace = AgentTrace(
            goal=f"Score job fit and draft outreach email for {candidate_name}"
        )
        wall_start = time.monotonic()

        system_prompt = (
            "You are Orion, a job search agent. Given a job description and resume, "
            "you MUST use your tools in this order:\n"
            "1. parse_resume — extract structured info from the resume\n"
            "2. score_fit — score how well the resume matches the job\n"
            "3. draft_email — draft a cold outreach email using the fit summary\n\n"
            "Use ALL three tools. After all tools have been called, respond with a JSON "
            "summary object containing: fit_score, matched_skills, skill_gaps, email_draft."
        )

        user_message = (
            f"Job Description:\n{job_description}\n\n"
            f"Resume:\n{resume_text}\n\n"
            f"Candidate name: {candidate_name}\n"
            f"Email tone: {tone}"
        )

        messages = [{"role": "user", "content": user_message}]

        # Agentic loop — keeps running until the model stops calling tools
        step_index = 0
        while True:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system_prompt,
                tools=TOOLS,
                messages=messages,
            )

            # Collect tool use blocks from this response turn
            tool_uses = [b for b in response.content if b.type == "tool_use"]

            if not tool_uses:
                # Model is done — extract final text response
                final_text = next(
                    (b.text for b in response.content if b.type == "text"), "{}"
                )
                try:
                    clean = final_text.strip().lstrip("```json").rstrip("```").strip()
                    trace.final_output = json.loads(clean)
                except Exception:
                    trace.final_output = {"raw": final_text}
                break

            # Execute each tool call
            tool_results = []
            for tool_use in tool_uses:
                t_start = time.monotonic()
                try:
                    fn = TOOL_FN_MAP[tool_use.name]
                    # Tools are sync for now; wrap in executor for async compatibility
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: fn(**tool_use.input)
                    )
                    status = "success"
                except Exception as e:
                    result = {"error": str(e)}
                    status = "error"

                duration = (time.monotonic() - t_start) * 1000

                trace.steps.append(
                    AgentStep(
                        step_index=step_index,
                        tool_name=tool_use.name,
                        tool_input=tool_use.input,
                        tool_output=result,
                        duration_ms=round(duration, 2),
                        status=status,
                    )
                )
                step_index += 1

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": json.dumps(result),
                })

            # Append assistant response + tool results to message history
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

        trace.status = "complete"
        trace.total_duration_ms = round((time.monotonic() - wall_start) * 1000, 2)
        return trace
