# Backend — Claude Context

**Owner:** Piyush
**Stack:** FastAPI + Python

See root `CLAUDE.md` for shared project context and `../project.md` for the full product spec.

---

## Current State

Piyush's existing backend code (`app.py`, `elo.py`, etc.) currently lives at the repo root.
The plan is to migrate it into this `backend/` directory in a follow-up PR.
Until then, treat the root-level Python files as the backend source.

---

## Filesystem Scope

**Only modify files within `backend/` (and root Python files until migrated) and `docs/api-contracts/`.** Never touch `mobile/`.
When adding or changing endpoints, **always update `docs/api-contracts/openapi.json` in the same PR.**

---

## Planned Structure (after migration)

```
backend/
├── app/
│   ├── main.py           # FastAPI app entry point
│   ├── routers/          # One file per resource
│   ├── models/           # DB models / Pydantic schemas
│   ├── auth/             # OAuth2, JWT, Google/Apple token verification
│   ├── elo/              # ELO calculation logic
│   └── database.py       # DB connection / session
├── requirements.txt
└── .env.example
```

---

## Conventions

- **Auth:** OAuth2 with JWT bearer tokens. Google + Apple ID tokens verified server-side.
- **ELO:** Each player starts at 1000 per community. K-factor 32. Doubles: team ELO = avg of members.
- **Error responses:** Always use RFC 7807 `application/problem+json` format:
  ```json
  { "type": "...", "title": "...", "status": 400, "detail": "..." }
  ```
- **Match validation:** submitted matches start as `pending_validation`. Auto-accepted after 24 hrs if not confirmed by opponent.
- **Breaking changes:** tag PR description with `[BREAKING]` — requires both Swapnil + Piyush to approve.
- **Spec drift:** CI runs `openapi-spec-validator` on every PR. Spec must always match the running app.

---

## Database

TBD — document here once chosen (PostgreSQL preferred). Include migration tool (Alembic recommended).

---

## Claude Agent

Use `.claude/agents/backend-dev.md` when working in this directory.
