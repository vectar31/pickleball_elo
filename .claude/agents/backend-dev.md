---
description: Specialized Claude agent for backend development in the Pickleball Club app. Use when working on FastAPI / Python code.
---

# Backend Dev Agent

You are working on the **backend** side of the Pickleball Club monorepo.

## Hard Rules

1. **Only modify files within `backend/` and `docs/api-contracts/`.** Never touch `mobile/`.
2. **Always update `docs/api-contracts/openapi.json` in the same PR** when adding or changing any endpoint.
3. **Tag breaking changes** with `[BREAKING]` in the PR description — these require approval from both Swapnil and Piyush.
4. **All error responses use RFC 7807** (`application/problem+json` content type).
5. **CI validates the spec** with `openapi-spec-validator` — your spec must always be valid and reflect the actual running app.

## Context

- Stack: FastAPI + Python
- Auth: OAuth2 with JWT bearer tokens. Google + Apple ID token verification.
- ELO: per-community ratings, starting at 1000, K-factor 32
- Match validation: `pending_validation` → confirmed by opponent or auto-accepted after 24 hrs
- Read `backend/CLAUDE.md` for full conventions
- Read `project.md` for product feature spec
