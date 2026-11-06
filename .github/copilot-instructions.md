# Aegis: Fault-Tolerant LLM Gateway & Semantic Cache

## What this project is
Async Python reverse proxy (FastAPI + Uvicorn) between client apps and LLM providers.
Exposes an OpenAI-compatible POST /v1/chat/completions endpoint (drop-in replacement).
MCA mini-project and CV portfolio piece. Headline: "Semantic cache with measured
false-positive control and a fault-tolerant LLM gateway."

## Tech stack (do not substitute without asking)
Python 3.12, uv, FastAPI, Uvicorn, HTTPX (async), Pydantic v2, FastEmbed (ONNX,
BAAI/bge-small-en-v1.5 or all-MiniLM-L6-v2), Qdrant (embedded in Phase 1-2, server mode in
Phase 3), Redis (Phase 3), a small quantized NLI cross-encoder (ONNX), pytest +
pytest-asyncio, Locust, Ruff, Mypy (strict), Docker Compose, Prometheus + Grafana (Phase 4).

## Architecture rules
1. Semantic cache = 3 stages: (1) cosine vector retrieval on USER CONTENT ONLY, with exact-match
   payload filters; (2) Stage-0 regex guard comparing numbers and negations; (3) bidirectional
   NLI check (A->B and B->A both must be entailment) on only the top 1-2 candidates.
2. Cache key / payload filters: system prompt hash, tenant ID, embedding model version, target
   LLM, temperature, max_tokens, tools, response_format. V1 caches ONLY single-turn,
   low-temperature (deterministic) requests. Multi-turn requests bypass the cache.
3. Circuit breaker FSM: CLOSED -> OPEN -> HALF-OPEN -> CLOSED. 5xx and timeouts count as
   failures. 429 is NOT a breaker failure: handle via exponential backoff with jitter and
   Retry-After. A full retry chain counts as ONE failure. The clock must be injectable
   (fake clock) for deterministic tests.
4. OPEN state = Degraded Mode: serve stale (post-TTL) cache entries using the SAME strict NLI
   threshold, add header `X-Aegis-Degraded: true`. No stale match -> 503 with that header.
   Lowering the threshold is opt-in config only.
5. Single-flight: per-worker request coalescing with asyncio.Event, keyed by exact strict
   cache key + identical payload (never semantic). Streaming requests are excluded.
6. Fail-open: if Redis or Qdrant is down, bypass cache/locking and call upstream directly;
   breaker falls back to local in-memory FSM.
7. SSE streaming: pass-through on miss while assembling chunks; if client disconnects
   mid-stream, do NOT cache the partial response; cache hits replay in SSE format.
8. Mock upstream server injects 429s, 500s, timeouts, and mid-stream disconnects.

## Engineering rules
- Work PHASE BY PHASE. Never start a later phase unless I say so.
- Write tests alongside code. Circuit breaker tests must use a fake clock.
- Full type hints, Mypy strict clean, Ruff clean. Small modules with single responsibility.
- Never fabricate benchmark numbers, dataset results, or latencies. All metrics must come from
  scripts I actually run. Put placeholders like `TODO: fill after running` in docs.
- I must be able to defend every line in a viva. After each task, give a short plain-English
  explanation of what you built and WHY, plus 2-3 likely viva questions on that code.
- Prefer simple and readable over clever. No unnecessary dependencies.
- Commit-sized steps: after each step, list files changed and how to run/test them.

## Directory layout
aegis/{api,core,engine,providers,storage}/, benchmarks/, tests/, notebooks/, docs/,
docker-compose.yml, Dockerfile, pyproject.toml, README.md