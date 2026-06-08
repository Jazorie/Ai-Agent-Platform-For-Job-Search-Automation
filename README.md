# Orion — AI Agent Platform

A general-purpose AI agent platform, demoed as a job search pipeline.
The architecture is domain-agnostic: swappable tool registry, full trace
observability, and persistent memory.

## Architecture

```
Input → Planner (Claude) → Tool Registry → Executor → Trace Store → Output
```

## Roadmap

| Phase | Focus |
|-------|-------|
| 1 | Core agent loop, tool registry, FastAPI |
| 2 | Dynamic tool registry (plug-and-play tools) |
| 3 | Memory layer — ChromaDB vector store |
| 4 | Failure recovery + retry logic |
| 5 | Observability dashboard — trace visualization |
| 6 | Docker + CI/CD |
