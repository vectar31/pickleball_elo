# Bug: `/stats/player/{name}` and `/stats/singles` return 500

**Reported:** 2026-03-21
**Reporter:** Swapnil (mobile)
**Severity:** Critical — profile screen fails to load for all users
**Status:** Open

---

## Symptom

The mobile profile screen shows "Failed to load profile." for every user.

Reproducing with curl confirms the backend is returning HTTP 500:

```bash
curl https://pickleball-elo.fly.dev/stats/player/Akshat
# → Internal Server Error

curl https://pickleball-elo.fly.dev/stats/singles
# → Internal Server Error
```

---

## Root Cause

`database.get_all_players()` was updated in the "new api" commit to return full player dicts:

```python
# What it returns now
[{"id": "uuid", "name": "Akshat", "display_name": "Akshat", ...}, ...]
```

But in `backend/app/routers/stats.py`, the result is passed directly to `compute_ratings_from_data()`:

```python
players = database.get_all_players()  # ← list of dicts, not names
singles_ratings, singles_history = compute_ratings_from_data(accepted_singles, players)
```

Inside `elo/engine.py`, `compute_ratings_from_data` does this on the first line:

```python
ratings = {p: DEFAULT_RATING for p in players}
```

When `p` is a `dict` (unhashable type), Python immediately raises:

```
TypeError: unhashable type: 'dict'
```

FastAPI catches this unhandled exception and returns **HTTP 500**.

---

## Affected Endpoints

| Endpoint | File | Line | Impact |
|---|---|---|---|
| `GET /stats/singles` | `routers/stats.py` | ~61 | Crashes — 500 |
| `GET /stats/player/{name}` | `routers/stats.py` | ~142 | Crashes — 500 |
| `GET /stats/doubles` | `routers/stats.py` | ~100 | Does **not** crash — uses `compute_doubles_ratings_from_data` which doesn't take a players list |

---

## Fix

Two lines in `backend/app/routers/stats.py`. Both need the same change — extract names from the dicts before passing to the ELO engine:

```python
# BEFORE (broken)
players = database.get_all_players()

# AFTER (fixed)
players = [p["name"] for p in database.get_all_players()]
```

Apply this in two places:

**1. `get_singles_stats` (~line 61):**

```python
@router.get("/singles")
def get_singles_stats():
    database.auto_accept_expired(hours=24)
    accepted = _accepted(database.get_singles_matches())
    players = [p["name"] for p in database.get_all_players()]  # ← fix here
    ratings, _ = compute_ratings_from_data(accepted, players)
    ...
```

**2. `get_player_stats` (~line 142):**

```python
@router.get("/player/{name}")
def get_player_stats(name: str):
    ...
    players = [p["name"] for p in database.get_all_players()]  # ← fix here
    singles_ratings, singles_history = compute_ratings_from_data(accepted_singles, players)
    ...
```

---

## How to Verify Fix

```bash
# After deploying fix to Fly.io:
curl https://pickleball-elo.fly.dev/stats/player/Akshat
# Should return JSON: { "player": "Akshat", "singles": { "elo": ..., "wins": ..., ... }, "doubles": { ... } }

curl https://pickleball-elo.fly.dev/stats/singles
# Should return array of player stats
```

---

## Notes

- The `GET /stats/doubles` endpoint is **not affected** because it calls `compute_doubles_ratings_from_data` which does not take a players list argument.
- The `players.py` router's `_player_stats` helper is **not affected** — it already extracts names correctly via a separate query path.
- Mobile code is correct and waiting on this backend fix to make the profile screen functional.
