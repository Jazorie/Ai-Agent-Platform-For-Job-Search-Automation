"""
Quick CLI test — run the agent without starting the API server.
Usage: python run_test.py
"""

import asyncio
import json
import os
from dotenv import load_dotenv

load_dotenv()

from agent.core import OrionAgent

# ── Sample data ───────────────────────────────────────────────────────────────

SAMPLE_JOB = """
Software Engineer, Backend Infrastructure — Anthropic

We're looking for a backend engineer to help build the systems that power Claude.
You'll work on high-throughput APIs, distributed job queues, and internal tooling
used by our research and product teams.

Requirements:
- 2+ years of backend engineering experience
- Proficiency in Python (FastAPI or Django preferred)
- Experience with PostgreSQL and Redis
- Familiarity with Docker and Kubernetes
- Experience with async Python (asyncio, aiohttp)

Nice to have:
- Experience with LLM APIs or ML infrastructure
- Contributions to open source projects
- Experience with AWS or GCP

Responsibilities:
- Design and maintain high-availability APIs serving millions of requests
- Build internal tooling for ML training pipelines
- Collaborate with research engineers on system design
"""

SAMPLE_RESUME = """
Jasmine Chen
jasmine@email.com | github.com/Jazorie | linkedin.com/in/jasminec

EDUCATION
California State University, Fullerton — B.S. Computer Science (Expected May 2028)
GPA: 3.7

SKILLS
Languages: Python, JavaScript, TypeScript, SQL, Java
Frameworks: React, FastAPI, Next.js, SQLAlchemy
Tools: Docker, Git, PostgreSQL, Redis, Vite, Figma
Concepts: REST APIs, async Python, JWT auth, vector databases

EXPERIENCE
ACM CSUF — Design Officer (Sep 2024 – Present)
- Led UI/UX workshops for 80+ members, created Figma design systems

Theta Tau Phi Epsilon — Web Developer (Jan 2024 – Present)
- Rebuilt chapter website with Next.js, reducing load time by 40%

JPMorgan Chase — Mentorship Program Participant (Summer 2024)
- Explored fintech product workflows through Big Brothers Big Sisters program

PROJECTS
TradeSim — Paper Trading Platform (github.com/trevorngo24/paper-trading-platform)
- Built React/Vite frontend + Python/FastAPI backend for simulated stock trading
- Implemented JWT auth, portfolio gain/loss endpoints, Finnhub stock search API
- Deployed with Docker, PostgreSQL via Railway

Satellite Tracker 2026 — Next.js 3D globe (github.com/Jazorie/satellite-tracker)
- Real-time satellite visualization using Space-Track.org TLE data
- Integrated Claude API for natural language orbital queries
- Resolved SSL/TLS errors, managed Git branching across 3 contributors

Growmicron — IoT Hydroponics System
- Raspberry Pi sensor network with FastAPI backend, PostgreSQL storage
- Real-time monitoring dashboard for nutrient and pH levels
"""


async def main():
    print("🚀 Running Orion Agent — Phase 1 Test\n")
    print("=" * 60)

    agent = OrionAgent()
    trace = await agent.run(
        job_description=SAMPLE_JOB,
        resume_text=SAMPLE_RESUME,
        candidate_name="Jasmine",
        tone="conversational",
    )

    print(f"Run ID:     {trace.run_id}")
    print(f"Status:     {trace.status}")
    print(f"Duration:   {trace.total_duration_ms}ms\n")

    print("── Steps ──────────────────────────────────────────────")
    for step in trace.steps:
        icon = "✅" if step.status == "success" else "❌"
        print(f"  {icon} [{step.step_index}] {step.tool_name} ({step.duration_ms}ms)")

    print("\n── Final Output ────────────────────────────────────────")
    print(json.dumps(trace.final_output, indent=2))

    if "fit_score" in trace.final_output:
        score = trace.final_output["fit_score"]
        bar = "█" * (score // 5) + "░" * (20 - score // 5)
        print(f"\n  Fit Score: [{bar}] {score}/100")

    if "email_draft" in trace.final_output:
        email = trace.final_output["email_draft"]
        if isinstance(email, dict):
            print(f"\n── Draft Email ─────────────────────────────────────────")
            print(f"  Subject: {email.get('subject', '')}")
            print(f"\n{email.get('body', '')}")


if __name__ == "__main__":
    asyncio.run(main())
