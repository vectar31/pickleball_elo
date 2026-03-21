# Pickleball Club — Shared Claude Context

This is the root Claude context for the Pickleball Club monorepo.
**Swapnil** owns `mobile/` (React Native + Expo + TypeScript).
**Piyush** owns `backend/` (FastAPI + Python).

See `project.md` for the full product vision and feature spec.

---

## Directory Overview

```
pickleball/
├── mobile/          # React Native + Expo (Swapnil)
├── backend/         # FastAPI + Python (Piyush)
├── docs/api-contracts/openapi.json  # Source of truth for API shape
├── .claude/agents/  # Specialized Claude agents per domain
└── Makefile         # Unified task runner
```

---

## Branching Convention

- `main` — stable, always deployable
- `mobile/<feature>` — mobile feature branches
- `backend/<feature>` — backend feature branches
- PRs: mobile PRs only touch `mobile/`; backend PRs only touch `backend/`
- API changes: backend PRs must also update `docs/api-contracts/openapi.json`

---

## Running the Stack

```bash
make dev-mobile    # Expo dev server
make dev-backend   # FastAPI with hot reload
make dev           # Both
```

---

## API Contract Workflow

1. Piyush adds/changes an endpoint → updates `docs/api-contracts/openapi.json` in same PR
2. Breaking changes tagged `[BREAKING]` in PR description — requires both devs to approve
3. After merge, run `make gen-api-client` to regenerate the typed client in `mobile/services/api/`
4. Error responses use RFC 7807 (`application/problem+json`) — mobile handles this shape uniformly

**Spec arbitration:** if mobile and backend agents disagree on API shape, the `docs/api-contracts/openapi.json` is authoritative. Escalate to human review if disputed.

---

## Domain Context

- `mobile/CLAUDE.md` — Expo Router, component patterns, state, API client, testing
- `backend/CLAUDE.md` — FastAPI conventions, auth, ELO logic, DB, error format

## Claude Agents

- `.claude/agents/mobile-dev.md` — use when working in `mobile/`
- `.claude/agents/backend-dev.md` — use when working in `backend/`
