from fastapi import APIRouter

from app import database
from app.elo.engine import compute_ratings_from_data, compute_doubles_ratings_from_data

router = APIRouter(prefix="/ratings", tags=["ratings"])


def _accepted(matches):
    return [m for m in matches if m.get("status", "accepted") == "accepted"]


@router.get("/singles")
def get_singles_ratings():
    database.auto_accept_expired(hours=24)
    matches = _accepted(database.get_singles_matches())
    players = database.get_all_players()
    ratings, _ = compute_ratings_from_data(matches, players)
    return sorted([{"player": p, "rating": r} for p, r in ratings.items()], key=lambda x: x["rating"], reverse=True)


@router.get("/doubles")
def get_doubles_ratings():
    database.auto_accept_expired(hours=24)
    matches = _accepted(database.get_doubles_matches())
    ratings, _ = compute_doubles_ratings_from_data(matches)
    return sorted([{"player": p, "rating": r} for p, r in ratings.items()], key=lambda x: x["rating"], reverse=True)
