# Backend API PRD — v1 Feature Set

**Author:** Swapnil Tiwari (mobile)
**For:** Piyush (backend)
**Date:** 2026-03-21
**Status:** Ready for implementation

---

## Context

The mobile app needs a set of backend APIs to power three key screens:
- **Home** — recent game activity feed
- **Community** — ELO leaderboard (singles)
- **Profile** — match history + pending match confirmations

The existing v1 API (`pickleball-elo.fly.dev`) has raw match/player/rating endpoints but lacks:
- Player profile objects (structured data beyond a name string)
- Pending match queue (maker-checker flow)
- Community-scoped leaderboards
- Per-player match history

This PRD specifies the new and updated endpoints needed. **All changes must be reflected in `docs/api-contracts/openapi.json` in the same PR.**

---

## Data Models

### `Player`
```json
{
  "id": "string (uuid)",
  "name": "string",
  "display_name": "string",
  "location": "string | null",
  "photo_url": "string | null",
  "created_at": "ISO 8601"
}
```

### `Match` (shared shape for singles and doubles)
```json
{
  "id": "string (uuid)",
  "format": "singles | doubles",
  "status": "pending_validation | accepted | disputed",
  "community_id": "string (uuid)",
  "submitted_by": "string (player id)",
  "played_at": "ISO 8601",
  "created_at": "ISO 8601",
  "score": {
    "team1": [11, 11, 9],
    "team2": [8, 7, 11]
  },
  "team1": ["player_id"],
  "team2": ["player_id"]
}
```
> For singles, each team array has exactly 1 player id.
> Score is an array of per-game points (best of N).

### `EloEntry`
```json
{
  "player_id": "string",
  "player_name": "string",
  "photo_url": "string | null",
  "elo": 1234,
  "delta": 12,
  "rank": 1,
  "wins": 10,
  "losses": 4
}
```

### `PendingMatch`
```json
{
  "match_id": "string (uuid)",
  "format": "singles | doubles",
  "community_id": "string (uuid)",
  "submitted_by_name": "string",
  "played_at": "ISO 8601",
  "expires_at": "ISO 8601",
  "score": { ... },
  "team1": ["player_id"],
  "team2": ["player_id"]
}
```

### Error shape (RFC 7807)
```json
{
  "type": "string (URI)",
  "title": "string",
  "status": 400,
  "detail": "string"
}
```
Content-Type: `application/problem+json`

---

## Endpoints Required

### 1. Player Profile

**GET `/players/{player_id}`**

Returns structured player information.

**Response 200:**
```json
{
  "player": { ...Player },
  "stats": {
    "singles": { "wins": 10, "losses": 4, "elo": 1234 },
    "doubles": { "wins": 6, "losses": 2, "elo": 1180 }
  }
}
```

**GET `/players/me`**

Same shape as above but for the authenticated user. Requires `Authorization: Bearer <token>`.

---

### 2. Singles Match List

**GET `/matches/singles`**

Returns all singles matches (paginated). Supports optional filters.

**Query params:**
| Param | Type | Description |
|---|---|---|
| `player_id` | string | Filter to matches involving this player |
| `community_id` | string | Filter by community |
| `status` | `pending_validation \| accepted \| disputed` | Filter by status |
| `limit` | int (default 20, max 100) | Page size |
| `cursor` | string | Cursor for next page |

**Response 200:**
```json
{
  "items": [ ...Match[] ],
  "next_cursor": "string | null"
}
```

---

### 3. Doubles Match List

**GET `/matches/doubles`**

Same shape and query params as `/matches/singles`.

---

### 4. Pending Matches (Maker-Checker Queue)

**GET `/matches/pending`**

Returns all matches awaiting confirmation **by the authenticated user**. This is the maker-checker queue shown on the Profile screen.

Requires `Authorization: Bearer <token>`.

**Response 200:**
```json
{
  "items": [ ...PendingMatch[] ]
}
```

---

**POST `/matches/{match_id}/confirm`**

Authenticated user confirms a pending match.

**Response 200:**
```json
{ "match_id": "string", "status": "accepted" }
```

**POST `/matches/{match_id}/dispute`**

Authenticated user disputes a pending match.

**Request body:**
```json
{ "reason": "string (optional)" }
```

**Response 200:**
```json
{ "match_id": "string", "status": "disputed" }
```

---

### 5. ELO Leaderboard

**GET `/communities/{community_id}/leaderboard`**

Returns the ELO leaderboard for a community, scoped to a format.

**Query params:**
| Param | Type | Description |
|---|---|---|
| `format` | `singles \| doubles` | Required. Which ELO ranking to show. |
| `limit` | int (default 50) | Page size |
| `cursor` | string | Cursor-based pagination |

**Response 200:**
```json
{
  "community_id": "string",
  "format": "singles",
  "items": [ ...EloEntry[] ],
  "next_cursor": "string | null"
}
```

> `delta` = ELO change in last 7 days. Used to show up/down arrows in the UI.

---

### 6. Recent Activity Feed (Home Screen)

**GET `/feed`**

Returns recent match activity across all communities the authenticated user belongs to. Powers the home screen feed.

Requires `Authorization: Bearer <token>`.

**Query params:**
| Param | Type | Description |
|---|---|---|
| `limit` | int (default 20) | Page size |
| `cursor` | string | Cursor-based pagination |

**Response 200:**
```json
{
  "items": [
    {
      "match": { ...Match },
      "community_name": "string",
      "players": { "<player_id>": { ...Player } }
    }
  ],
  "next_cursor": "string | null"
}
```

> Include a `players` map in each item so the mobile client can render names/photos without extra requests.

---

## Screen → Endpoint Mapping

| Screen | Endpoint(s) |
|---|---|
| Home — recent activity | `GET /feed` |
| Community — ELO leaderboard (singles) | `GET /communities/{id}/leaderboard?format=singles` |
| Profile — my stats | `GET /players/me` |
| Profile — recent singles matches | `GET /matches/singles?player_id=me` |
| Profile — recent doubles matches | `GET /matches/doubles?player_id=me` |
| Profile — pending confirmations | `GET /matches/pending` |
| Profile — confirm a match | `POST /matches/{id}/confirm` |
| Profile — dispute a match | `POST /matches/{id}/dispute` |

---

## Auth

All endpoints that return user-specific data (`/players/me`, `/matches/pending`, `/feed`, `/matches/{id}/confirm`, `/matches/{id}/dispute`) require a valid JWT in the `Authorization: Bearer` header.

Return `401 application/problem+json` with `status: 401` for missing/expired tokens.

---

## Pagination Convention

Use cursor-based pagination for all list endpoints (not offset). Include `next_cursor: null` when there are no more pages.

---

## Notes for Piyush

- `delta` on `EloEntry` can be computed as `current_elo - elo_7_days_ago`. If < 7 days of history, default to `0`.
- The `expires_at` on `PendingMatch` = `played_at + 24h`. Auto-accept cron job should flip status to `accepted` after expiry.
- The `players` map in `/feed` responses should only include players who appear in that match (avoid over-fetching).
- All list endpoints should be usable without auth except where noted, but authenticated requests can return personalized ordering/highlights in a later iteration.
- Once endpoints are live, update `docs/api-contracts/openapi.json` and tag Swapnil — mobile will regenerate the typed client with `make gen-api-client`.
