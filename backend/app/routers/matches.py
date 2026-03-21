from fastapi import APIRouter, Depends, HTTPException, status

from app import database
from app.auth.jwt import get_current_user
from app.models.schemas import SinglesMatchInput, DoublesMatchInput

router = APIRouter(prefix="/matches", tags=["matches"])


def _run_auto_accept():
    database.auto_accept_expired(hours=24)


# ── Singles ───────────────────────────────────────────────────────────────────

@router.get("/singles")
def get_singles_matches():
    return database.get_singles_matches()


@router.post("/singles", status_code=status.HTTP_201_CREATED)
def add_singles_match(match: SinglesMatchInput, current_user: str = Depends(get_current_user)):
    if match.player1 == match.player2:
        raise HTTPException(status_code=400, detail="Players must be different")
    if match.score1 == match.score2:
        raise HTTPException(status_code=400, detail="Ties are not allowed")
    if current_user not in (match.player1, match.player2):
        raise HTTPException(status_code=403, detail="You can only submit matches you played in")
    database.insert_singles_match(match.date, match.player1, match.player2, match.score1, match.score2, submitted_by=current_user)
    return {"message": "Match submitted — awaiting opponent confirmation"}


@router.post("/singles/{match_id}/confirm")
def confirm_singles_match(match_id: int, current_user: str = Depends(get_current_user)):
    match = database.get_singles_match_by_id(match_id)
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    if match["status"] != "pending_validation":
        raise HTTPException(status_code=400, detail=f"Match is already {match['status']}")
    if current_user == match["submitted_by"]:
        raise HTTPException(status_code=403, detail="You cannot confirm your own match")
    if current_user not in (match["player1"], match["player2"]):
        raise HTTPException(status_code=403, detail="You are not a participant in this match")
    database.update_singles_match_status(match_id, "accepted")
    return {"message": "Match confirmed"}


@router.post("/singles/{match_id}/dispute")
def dispute_singles_match(match_id: int, current_user: str = Depends(get_current_user)):
    match = database.get_singles_match_by_id(match_id)
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    if match["status"] != "pending_validation":
        raise HTTPException(status_code=400, detail=f"Match is already {match['status']}")
    if current_user == match["submitted_by"]:
        raise HTTPException(status_code=403, detail="You cannot dispute your own match")
    if current_user not in (match["player1"], match["player2"]):
        raise HTTPException(status_code=403, detail="You are not a participant in this match")
    database.update_singles_match_status(match_id, "disputed")
    return {"message": "Match disputed — flagged for review"}


# ── Doubles ───────────────────────────────────────────────────────────────────

@router.get("/doubles")
def get_doubles_matches():
    return database.get_doubles_matches()


@router.post("/doubles", status_code=status.HTTP_201_CREATED)
def add_doubles_match(match: DoublesMatchInput, current_user: str = Depends(get_current_user)):
    if set(match.team1) & set(match.team2):
        raise HTTPException(status_code=400, detail="A player cannot be on both teams")
    if match.score1 == match.score2:
        raise HTTPException(status_code=400, detail="Ties are not allowed")
    if current_user not in match.team1 + match.team2:
        raise HTTPException(status_code=403, detail="You can only submit matches you played in")
    database.insert_doubles_match(match.date, match.team1, match.team2, match.score1, match.score2, submitted_by=current_user)
    return {"message": "Match submitted — awaiting opponent confirmation"}


@router.post("/doubles/{match_id}/confirm")
def confirm_doubles_match(match_id: int, current_user: str = Depends(get_current_user)):
    match = database.get_doubles_match_by_id(match_id)
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    if match["status"] != "pending_validation":
        raise HTTPException(status_code=400, detail=f"Match is already {match['status']}")
    if current_user == match["submitted_by"]:
        raise HTTPException(status_code=403, detail="You cannot confirm your own match")
    submitter_team = match["team1"] if match["submitted_by"] in match["team1"] else match["team2"]
    opponent_team = match["team2"] if submitter_team == match["team1"] else match["team1"]
    if current_user not in opponent_team:
        raise HTTPException(status_code=403, detail="You are not on the opposing team")
    database.update_doubles_match_status(match_id, "accepted")
    return {"message": "Match confirmed"}


@router.post("/doubles/{match_id}/dispute")
def dispute_doubles_match(match_id: int, current_user: str = Depends(get_current_user)):
    match = database.get_doubles_match_by_id(match_id)
    if not match:
        raise HTTPException(status_code=404, detail="Match not found")
    if match["status"] != "pending_validation":
        raise HTTPException(status_code=400, detail=f"Match is already {match['status']}")
    if current_user == match["submitted_by"]:
        raise HTTPException(status_code=403, detail="You cannot dispute your own match")
    submitter_team = match["team1"] if match["submitted_by"] in match["team1"] else match["team2"]
    opponent_team = match["team2"] if submitter_team == match["team1"] else match["team1"]
    if current_user not in opponent_team:
        raise HTTPException(status_code=403, detail="You are not on the opposing team")
    database.update_doubles_match_status(match_id, "disputed")
    return {"message": "Match disputed — flagged for review"}


# ── Pending ───────────────────────────────────────────────────────────────────

@router.get("/pending")
def get_pending_matches(current_user: str = Depends(get_current_user)):
    _run_auto_accept()
    return database.get_pending_for_player(current_user)
