from typing import Optional

from fastapi import APIRouter, Depends, Query

from app import database
from app.auth.jwt import get_current_user

router = APIRouter(prefix="/feed", tags=["feed"])


@router.get("")
def get_feed(
    limit: int = Query(20, le=100),
    cursor: Optional[str] = Query(None),
    current_user: str = Depends(get_current_user),
):
    database.auto_accept_expired()
    items, next_cursor = database.get_feed(limit=limit, cursor=cursor)

    # Build players map — only players who appear in this page
    player_names = set()
    for item in items:
        player_names.update(item["team1"] + item["team2"])

    all_players = {p["name"]: p for p in database.get_all_players()}
    players_map = {name: all_players[name] for name in player_names if name in all_players}

    return {
        "items": [
            {
                "match": item,
                "community_name": database.COMMUNITY_NAME,
                "players": {
                    name: players_map[name]
                    for name in item["team1"] + item["team2"]
                    if name in players_map
                },
            }
            for item in items
        ],
        "next_cursor": next_cursor,
    }
