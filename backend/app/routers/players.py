from collections import defaultdict

from fastapi import APIRouter, Depends, HTTPException

from app import database
from app.auth.jwt import get_current_user
from app.elo.engine import compute_ratings_from_data, compute_doubles_ratings_from_data

router = APIRouter(prefix="/players", tags=["players"])


def _accepted(matches):
    return [m for m in matches if m.get("status", "accepted") == "accepted"]


def _player_stats(name: str) -> dict:
    database.auto_accept_expired()
    singles = _accepted(database.get_singles_matches(player_name=name, limit=10000))
    doubles = _accepted(database.get_doubles_matches(player_name=name, limit=10000))
    all_singles = _accepted(database.get_singles_matches(limit=10000))
    all_doubles = _accepted(database.get_doubles_matches(limit=10000))
    players = [p["name"] for p in database.get_all_players()]

    singles_ratings, _ = compute_ratings_from_data(all_singles, players)
    doubles_ratings, _ = compute_doubles_ratings_from_data(all_doubles)

    s_wins = sum(
        1 for m in singles
        if (m["player1"] == name and m["score1"] > m["score2"]) or
           (m["player2"] == name and m["score2"] > m["score1"])
    )
    s_losses = len(singles) - s_wins

    d_wins = sum(
        1 for m in doubles
        if (name in m["team1"] and m["score1"] > m["score2"]) or
           (name in m["team2"] and m["score2"] > m["score1"])
    )
    d_losses = len(doubles) - d_wins

    return {
        "singles": {"wins": s_wins, "losses": s_losses, "elo": singles_ratings.get(name, 1000)},
        "doubles": {"wins": d_wins, "losses": d_losses, "elo": doubles_ratings.get(name, 1000)},
    }


@router.get("")
def get_players():
    return {"players": database.get_all_players()}


@router.get("/me")
def get_me(current_user: str = Depends(get_current_user)):
    player = database.get_player_by_id(current_user)
    if not player:
        raise HTTPException(status_code=404, detail="Player not found")
    return {"player": player, "stats": _player_stats(current_user)}


@router.get("/{player_id}")
def get_player(player_id: str):
    player = database.get_player_by_id(player_id)
    if not player:
        raise HTTPException(status_code=404, detail="Player not found")
    return {"player": player, "stats": _player_stats(player["name"])}
