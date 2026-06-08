# Orion — AI Agent Platform

A general-purpose AI agent platform, demoed as a job search pipeline.
The architecture is domain-agnostic: swappable tool registry, full trace
observability, and persistent memory (Phase 3).

## Architecture

```
Input → Planner (Claude) → Tool Registry → Executor → Trace Store → Output
```

## Phase 1 — What's here

- **Agent core** (`agent/core.py`) — agentic loop with tool calling, trace recording
- **Tools** (`agent/tools/`) — fit scorer, email drafter, resume parser
- **API** (`api/server.py`) — FastAPI with `/analyze`, `/traces` endpoints

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your API key
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# 3. Run the CLI test (no server needed)
python run_test.py

# 4. Start the API server
python main.py
# → http://localhost:8000/docs
```

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/analyze` | Run the agent on a job + resume |
| GET | `/traces` | List all agent run summaries |
| GET | `/traces/{run_id}` | Full trace for a specific run |
| GET | `/health` | Health check |

## Example request

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "job_description": "...",
    "resume_text": "...",
    "candidate_name": "Jasmine",
    "tone": "conversational"
  }'
```

## Roadmap

| Phase | Focus |
|-------|-------|
| ✅ 1 | Core agent loop, tool registry, FastAPI |
| 2 | Dynamic tool registry (plug-and-play tools) |
| 3 | Memory layer — ChromaDB vector store |
| 4 | Failure recovery + retry logic |
| 5 | Observability dashboard — trace visualization |
| 6 | Docker + CI/CD |
