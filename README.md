# Pickleball Club

A cross-platform mobile app for closed pickleball communities — track matches, compete on ELO leaderboards, and build community around the game.

**Mobile:** React Native + Expo + TypeScript (Swapnil)
**Backend:** FastAPI + Python (Piyush)

---

## Quickstart

```bash
# Mobile
make dev-mobile

# Backend
make dev-backend

# Both
make dev
```

## API Client

After any backend spec update:
```bash
make gen-api-client
```

## Structure

```
mobile/     React Native app (Expo Router)
backend/    FastAPI backend
docs/       API contracts, architecture docs
.claude/    Claude Code agent configs
```

See `project.md` for product spec.
See `CLAUDE.md` for Claude Code collaboration context.
