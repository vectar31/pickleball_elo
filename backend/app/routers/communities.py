from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app import database
from app.elo.engine import compute_ratings_from_data, compute_doubles_ratings_from_data

router = APIRouter(prefix="/communities", tags=["communities"])

_SEVEN_DAYS_AGO = lambda: (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")


def _accepted(matches):
    return [m for m in matches if m.get("status", "accepted") == "accepted"]


def _wins_losses_singles(matches, player_name):
    wins = sum(
        1 for m in matches
        if (m["player1"] == player_name and m["score1"] > m["score2"]) or
           (m["player2"] == player_name and m["score2"] > m["score1"])
    )
    return wins, len([m for m in matches if m["player1"] == player_name or m["player2"] == player_name]) - wins


def _wins_losses_doubles(matches, player_name):
    player_matches = [m for m in matches if player_name in m["team1"] or player_name in m["team2"]]
    wins = sum(
        1 for m in player_matches
        if (player_name in m["team1"] and m["score1"] > m["score2"]) or
           (player_name in m["team2"] and m["score2"] > m["score1"])
    )
    return wins, len(player_matches) - wins


@router.get("/{community_id}/leaderboard")
def get_leaderboard(
    community_id: str,
    format: str = Query(..., description="singles or doubles"),
    limit: int = Query(50, le=100),
    cursor: Optional[str] = Query(None),
):
    community = database.get_community(community_id)
    if not community:
        raise HTTPException(status_code=404, detail="Community not found")
    if format not in ("singles", "doubles"):
        raise HTTPException(status_code=400, detail="format must be 'singles' or 'doubles'")

    database.auto_accept_expired()
    cutoff = _SEVEN_DAYS_AGO()

    if format == "singles":
        all_matches = _accepted(database.get_singles_matches(limit=100000))
        recent_matches = [m for m in all_matches if m.get("date", "") >= cutoff]
        old_matches = [m for m in all_matches if m.get("date", "") < cutoff]

        players = [p["name"] for p in database.get_all_players()]
        current_ratings, _ = compute_ratings_from_data(all_matches, players)
        old_ratings, _ = compute_ratings_from_data(old_matches, players)

        all_players_db = {p["name"]: p for p in database.get_all_players()}

        # Only include players who have played
        active = set()
        for m in all_matches:
            active.update([m["player1"], m["player2"]])

        entries = []
        for name in active:
            current_elo = current_ratings.get(name, 1000)
            prior_elo = old_ratings.get(name, 1000)
            wins, losses = _wins_losses_singles(all_matches, name)
            p = all_players_db.get(name, {})
            entries.append({
                "player_id": p.get("id", name),
                "player_name": p.get("display_name", name),
                "photo_url": p.get("photo_url"),
                "elo": round(current_elo, 1),
                "delta": round(current_elo - prior_elo, 1),
                "wins": wins,
                "losses": losses,
            })

    else:  # doubles
        all_matches = _accepted(database.get_doubles_matches(limit=100000))
        old_matches = [m for m in all_matches if m.get("date", "") < cutoff]

        current_ratings, _ = compute_doubles_ratings_from_data(all_matches)
        old_ratings, _ = compute_doubles_ratings_from_data(old_matches)

        all_players_db = {p["name"]: p for p in database.get_all_players()}

        active = set()
        for m in all_matches:
            active.update(m["team1"] + m["team2"])

        entries = []
        for name in active:
            current_elo = current_ratings.get(name, 1000)
            prior_elo = old_ratings.get(name, 1000)
            wins, losses = _wins_losses_doubles(all_matches, name)
            p = all_players_db.get(name, {})
            entries.append({
                "player_id": p.get("id", name),
                "player_name": p.get("display_name", name),
                "photo_url": p.get("photo_url"),
                "elo": round(current_elo, 1),
                "delta": round(current_elo - prior_elo, 1),
                "wins": wins,
                "losses": losses,
            })

    # Sort by ELO and assign rank
    entries.sort(key=lambda x: x["elo"], reverse=True)

    # Cursor-based pagination by rank offset
    start = 0
    if cursor:
        try:
            start = int(cursor)
        except ValueError:
            pass

    page = entries[start: start + limit]
    for i, entry in enumerate(page):
        entry["rank"] = start + i + 1

    next_cursor = str(start + limit) if start + limit < len(entries) else None

    return {
        "community_id": community_id,
        "format": format,
        "items": page,
        "next_cursor": next_cursor,
    }
