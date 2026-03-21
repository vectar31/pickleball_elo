.PHONY: dev dev-mobile dev-backend lint lint-mobile lint-backend test test-mobile test-backend gen-api-client

# ── Development ──────────────────────────────────────────────────────────────

dev-mobile:
	cd mobile && npx expo start

dev-backend:
	cd backend && uvicorn app.main:app --reload

dev:
	make dev-backend & make dev-mobile

# ── Linting ───────────────────────────────────────────────────────────────────

lint-mobile:
	cd mobile && npx eslint . && npx tsc --noEmit

lint-backend:
	cd backend && ruff check . && mypy .

lint: lint-mobile lint-backend

# ── Testing ───────────────────────────────────────────────────────────────────

test-mobile:
	cd mobile && npx jest

test-backend:
	cd backend && pytest

test: test-mobile test-backend

# ── API Codegen ───────────────────────────────────────────────────────────────

# Regenerate typed TypeScript client from OpenAPI spec.
# Run after any backend spec update (docs/api-contracts/openapi.json).
gen-api-client:
	cd mobile && npx orval --config orval.config.ts
