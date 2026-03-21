from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app import database
from app.auth.jwt import get_current_user, get_admin_user
from app.models.schemas import SinglesMatchInput, DoublesMatchInput, DisputeInput

router = APIRouter(prefix="/matches", tags=["matches"])

_DEFAULT_LIMIT = 20
_MAX_LIMIT = 100


def _paginate(items: list, limit: int):
    """Return (page_items, next_cursor) where cursor is the last item's id."""
    has_more = len(items) > limit
    page = items[:limit]
    next_cursor = str(page[-1]["id"]) if has_more and page else None
    return page, next_cursor


# ── Singles ───────────────────────────────────────────────────────────────────

@router.get("/singles")
def get_singles_matches(
    player_id: Optional[str] = Query(None, description="Filter by player name or 'me'"),
    status: Optional[str] = Query(None),
    limit: int = Query(_DEFAULT_LIMIT, le=_MAX_LIMIT),
    cursor: Optional[int] = Query(None),
    current_user: Optional[str] = Depends(get_current_user.__wrapped__ if hasattr(get_current_user, '__wrapped__') else get_current_user),
):
    database.auto_accept_expired()
    player_name = current_user if player_id == "me" else player_id
    rows = database.get_singles_matches(status=status, player_name=player_name, limit=limit + 1, cursor=cursor)
    items, next_cursor = _paginate(rows, limit)
    return {"items": items, "next_cursor": next_cursor}


@router.post("/singles", status_code=status.HTTP_201_CREATED)
def add_singles_match(match: SinglesMatchInput, current_user: str = Depends(get_admin_user)):
    if match.player1 == match.player2:
        raise HTTPException(status_code=400, detail="Players must be different")
    if match.score1 == match.score2:
        raise HTTPException(status_code=400, detail="Ties are not allowed")
    database.insert_singles_match(match.date, match.player1, match.player2, match.score1, match.score2, submitted_by=current_user)
    return {"message": "Match submitted — awaiting opponent confirmation"}


# ── Doubles ───────────────────────────────────────────────────────────────────

@router.get("/doubles")
def get_doubles_matches(
    player_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(_DEFAULT_LIMIT, le=_MAX_LIMIT),
    cursor: Optional[int] = Query(None),
):
    database.auto_accept_expired()
    rows = database.get_doubles_matches(status=status, player_name=player_id, limit=limit + 1, cursor=cursor)
    items, next_cursor = _paginate(rows, limit)
    return {"items": items, "next_cursor": next_cursor}


@router.post("/doubles", status_code=status.HTTP_201_CREATED)
def add_doubles_match(match: DoublesMatchInput, current_user: str = Depends(get_admin_user)):
    if set(match.team1) & set(match.team2):
        raise HTTPException(status_code=400, detail="A player cannot be on both teams")
    if match.score1 == match.score2:
        raise HTTPException(status_code=400, detail="Ties are not allowed")
    database.insert_doubles_match(match.date, match.team1, match.team2, match.score1, match.score2, submitted_by=current_user)
    return {"message": "Match submitted — awaiting opponent confirmation"}


# ── Pending ───────────────────────────────────────────────────────────────────

@router.get("/pending")
def get_pending_matches(current_user: str = Depends(get_current_user)):
    database.auto_accept_expired()
    return {"items": database.get_pending_for_player(current_user)}


# ── Unified confirm / dispute ─────────────────────────────────────────────────

@router.post("/{match_id}/confirm")
def confirm_match(match_id: int, current_user: str = Depends(get_current_user)):
    match, fmt = database.get_any_match_by_id(match_id)
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    if match["status"] != "pending_validation":
        raise HTTPException(status_code=400, detail=f"Match is already {match['status']}")
    if current_user == match["submitted_by"]:
        raise HTTPException(status_code=403, detail="You cannot confirm your own match")

    if fmt == "singles":
        if current_user not in (match["player1"], match["player2"]):
            raise HTTPException(status_code=403, detail="You are not a participant in this match")
        database.update_singles_match_status(match_id, "accepted")
    else:
        submitter_team = match["team1"] if match["submitted_by"] in match["team1"] else match["team2"]
        opponent_team = match["team2"] if submitter_team == match["team1"] else match["team1"]
        if current_user not in opponent_team:
            raise HTTPException(status_code=403, detail="You are not on the opposing team")
        database.update_doubles_match_status(match_id, "accepted")

    return {"match_id": str(match_id), "status": "accepted"}


@router.post("/{match_id}/dispute")
def dispute_match(match_id: int, body: DisputeInput = DisputeInput(), current_user: str = Depends(get_current_user)):
    match, fmt = database.get_any_match_by_id(match_id)
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    if match["status"] != "pending_validation":
        raise HTTPException(status_code=400, detail=f"Match is already {match['status']}")
    if current_user == match["submitted_by"]:
        raise HTTPException(status_code=403, detail="You cannot dispute your own match")

    if fmt == "singles":
        if current_user not in (match["player1"], match["player2"]):
            raise HTTPException(status_code=403, detail="You are not a participant in this match")
        database.update_singles_match_status(match_id, "disputed")
    else:
        submitter_team = match["team1"] if match["submitted_by"] in match["team1"] else match["team2"]
        opponent_team = match["team2"] if submitter_team == match["team1"] else match["team1"]
        if current_user not in opponent_team:
            raise HTTPException(status_code=403, detail="You are not on the opposing team")
        database.update_doubles_match_status(match_id, "disputed")

    return {"match_id": str(match_id), "status": "disputed"}
