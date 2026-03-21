from collections import defaultdict
from datetime import datetime
from typing import List

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, field_validator

import auth
import database
from elo import compute_ratings_from_data, compute_doubles_ratings_from_data

app = FastAPI(title="Pickleball ELO API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    database.init_db()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _accepted(matches: list) -> list:
    """Filter to only accepted matches for ELO computation."""
    return [m for m in matches if m.get("status", "accepted") == "accepted"]


def _run_auto_accept():
    database.auto_accept_expired(hours=24)


# ── Schemas ───────────────────────────────────────────────────────────────────

class SinglesMatchInput(BaseModel):
    player1: str
    player2: str
    score1: int
    score2: int
    date: str = ""

    @field_validator("date", mode="before")
    @classmethod
    def default_date(cls, v):
        return v or datetime.today().strftime("%Y-%m-%d")


class DoublesMatchInput(BaseModel):
    team1: List[str]
    team2: List[str]
    score1: int
    score2: int
    date: str = ""

    @field_validator("date", mode="before")
    @classmethod
    def default_date(cls, v):
        return v or datetime.today().strftime("%Y-%m-%d")

    @field_validator("team1", "team2")
    @classmethod
    def two_players(cls, v):
        if len(v) != 2:
            raise ValueError("Each team must have exactly 2 players")
        return v


# ── Auth ──────────────────────────────────────────────────────────────────────

@app.post("/auth/login")
def login(form: OAuth2PasswordRequestForm = Depends()):
    row = database.get_player(form.username)
    if not row or not auth.verify_password(form.password, row["password_hash"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    token = auth.create_access_token(form.username)
    return {"access_token": token, "token_type": "bearer"}


# ── Players ───────────────────────────────────────────────────────────────────

@app.get("/players")
def get_players():
    return {"players": database.get_all_players()}


# ── Ratings ───────────────────────────────────────────────────────────────────

@app.get("/ratings/singles")
def get_singles_ratings():
    _run_auto_accept()
    matches = _accepted(database.get_singles_matches())
    players = database.get_all_players()
    ratings, _ = compute_ratings_from_data(matches, players)
    return sorted([{"player": p, "rating": r} for p, r in ratings.items()], key=lambda x: x["rating"], reverse=True)


@app.get("/ratings/doubles")
def get_doubles_ratings():
    _run_auto_accept()
    matches = _accepted(database.get_doubles_matches())
    ratings, _ = compute_doubles_ratings_from_data(matches)
    return sorted([{"player": p, "rating": r} for p, r in ratings.items()], key=lambda x: x["rating"], reverse=True)


# ── Matches ───────────────────────────────────────────────────────────────────

@app.get("/matches/singles")
def get_singles_matches():
    return database.get_singles_matches()


@app.post("/matches/singles", status_code=status.HTTP_201_CREATED)
def add_singles_match(match: SinglesMatchInput, current_user: str = Depends(auth.get_current_user)):
    if match.player1 == match.player2:
        raise HTTPException(status_code=400, detail="Players must be different")
    if match.score1 == match.score2:
        raise HTTPException(status_code=400, detail="Ties are not allowed")
    if current_user not in (match.player1, match.player2):
        raise HTTPException(status_code=403, detail="You can only submit matches you played in")
    database.insert_singles_match(match.date, match.player1, match.player2, match.score1, match.score2, submitted_by=current_user)
    return {"message": "Match submitted — awaiting opponent confirmation"}


@app.post("/matches/singles/{match_id}/confirm", status_code=status.HTTP_200_OK)
def confirm_singles_match(match_id: int, current_user: str = Depends(auth.get_current_user)):
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


@app.post("/matches/singles/{match_id}/dispute", status_code=status.HTTP_200_OK)
def dispute_singles_match(match_id: int, current_user: str = Depends(auth.get_current_user)):
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


@app.get("/matches/doubles")
def get_doubles_matches():
    return database.get_doubles_matches()


@app.post("/matches/doubles", status_code=status.HTTP_201_CREATED)
def add_doubles_match(match: DoublesMatchInput, current_user: str = Depends(auth.get_current_user)):
    all_players = match.team1 + match.team2
    if set(match.team1) & set(match.team2):
        raise HTTPException(status_code=400, detail="A player cannot be on both teams")
    if match.score1 == match.score2:
        raise HTTPException(status_code=400, detail="Ties are not allowed")
    if current_user not in all_players:
        raise HTTPException(status_code=403, detail="You can only submit matches you played in")
    database.insert_doubles_match(match.date, match.team1, match.team2, match.score1, match.score2, submitted_by=current_user)
    return {"message": "Match submitted — awaiting opponent confirmation"}


@app.post("/matches/doubles/{match_id}/confirm", status_code=status.HTTP_200_OK)
def confirm_doubles_match(match_id: int, current_user: str = Depends(auth.get_current_user)):
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


@app.post("/matches/doubles/{match_id}/dispute", status_code=status.HTTP_200_OK)
def dispute_doubles_match(match_id: int, current_user: str = Depends(auth.get_current_user)):
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


@app.get("/matches/pending")
def get_pending_matches(current_user: str = Depends(auth.get_current_user)):
    """Return all matches waiting for the current user's confirmation."""
    _run_auto_accept()
    return database.get_pending_for_player(current_user)


# ── Stats ─────────────────────────────────────────────────────────────────────

def _compute_singles_stats(matches):
    stats = defaultdict(lambda: {
        "wins": 0, "losses": 0, "points_won": 0, "points_lost": 0, "history": []
    })
    for m in matches:
        p1, p2, s1, s2 = m["player1"], m["player2"], m["score1"], m["score2"]
        winner, loser = (p1, p2) if s1 > s2 else (p2, p1)
        stats[winner]["wins"] += 1
        stats[loser]["losses"] += 1
        stats[p1]["points_won"] += s1
        stats[p1]["points_lost"] += s2
        stats[p2]["points_won"] += s2
        stats[p2]["points_lost"] += s1
        stats[p1]["history"].append("W" if p1 == winner else "L")
        stats[p2]["history"].append("W" if p2 == winner else "L")
    return stats


def _streak_info(history):
    if not history:
        return "", 0, 0
    last = history[-1]
    count = 0
    for r in reversed(history):
        if r == last:
            count += 1
        else:
            break
    current = f"{count}{last}"

    def max_streak(seq, target):
        best = cur = 0
        for r in seq:
            cur = cur + 1 if r == target else 0
            best = max(best, cur)
        return best

    return current, max_streak(history, "W"), max_streak(history, "L")


@app.get("/stats/singles")
def get_singles_stats():
    _run_auto_accept()
    all_matches = database.get_singles_matches()
    accepted = _accepted(all_matches)
    players = database.get_all_players()
    ratings, _ = compute_ratings_from_data(accepted, players)
    raw = _compute_singles_stats(accepted)

    result = []
    for player, data in raw.items():
        games = data["wins"] + data["losses"]
        if games == 0:
            continue
        current_streak, longest_win, longest_loss = _streak_info(data["history"])
        result.append({
            "player": player,
            "elo": ratings.get(player, 1000),
            "matches": games,
            "wins": data["wins"],
            "losses": data["losses"],
            "win_pct": round(data["wins"] / games * 100, 1),
            "avg_points_won": round(data["points_won"] / games, 1),
            "avg_points_lost": round(data["points_lost"] / games, 1),
            "current_streak": current_streak,
            "longest_win_streak": longest_win,
            "longest_loss_streak": longest_loss,
        })

    return sorted(result, key=lambda x: x["elo"], reverse=True)


@app.get("/stats/doubles")
def get_doubles_stats():
    _run_auto_accept()
    accepted = _accepted(database.get_doubles_matches())
    ratings, _ = compute_doubles_ratings_from_data(accepted)

    stats = defaultdict(lambda: {"wins": 0, "losses": 0, "points_won": 0, "points_lost": 0, "history": []})
    for m in accepted:
        t1, t2, s1, s2 = m["team1"], m["team2"], m["score1"], m["score2"]
        winners, losers = (t1, t2) if s1 > s2 else (t2, t1)
        for p in winners:
            stats[p]["wins"] += 1
            stats[p]["history"].append("W")
        for p in losers:
            stats[p]["losses"] += 1
            stats[p]["history"].append("L")
        for p in t1:
            stats[p]["points_won"] += s1
            stats[p]["points_lost"] += s2
        for p in t2:
            stats[p]["points_won"] += s2
            stats[p]["points_lost"] += s1

    result = []
    for player, data in stats.items():
        games = data["wins"] + data["losses"]
        if games == 0:
            continue
        current_streak, longest_win, longest_loss = _streak_info(data["history"])
        result.append({
            "player": player,
            "doubles_elo": ratings.get(player, 1000),
            "matches": games,
            "wins": data["wins"],
            "losses": data["losses"],
            "win_pct": round(data["wins"] / games * 100, 1),
            "avg_points_won": round(data["points_won"] / games, 1),
            "avg_points_lost": round(data["points_lost"] / games, 1),
            "current_streak": current_streak,
            "longest_win_streak": longest_win,
            "longest_loss_streak": longest_loss,
        })

    return sorted(result, key=lambda x: x["doubles_elo"], reverse=True)


@app.get("/stats/player/{name}")
def get_player_stats(name: str):
    if not database.player_exists(name):
        raise HTTPException(status_code=404, detail="Player not found")

    _run_auto_accept()
    accepted_singles = _accepted(database.get_singles_matches())
    accepted_doubles = _accepted(database.get_doubles_matches())
    players = database.get_all_players()

    singles_ratings, singles_history = compute_ratings_from_data(accepted_singles, players)
    doubles_ratings, doubles_history = compute_doubles_ratings_from_data(accepted_doubles)

    raw = _compute_singles_stats(accepted_singles)
    data = raw.get(name, {"wins": 0, "losses": 0, "points_won": 0, "points_lost": 0, "history": []})
    games = data["wins"] + data["losses"]
    current_streak, longest_win, longest_loss = _streak_info(data["history"])

    h2h = defaultdict(lambda: {"wins": 0, "losses": 0})
    for m in accepted_singles:
        if m["player1"] == name or m["player2"] == name:
            opp = m["player2"] if m["player1"] == name else m["player1"]
            won = (m["player1"] == name and m["score1"] > m["score2"]) or \
                  (m["player2"] == name and m["score2"] > m["score1"])
            if won:
                h2h[opp]["wins"] += 1
            else:
                h2h[opp]["losses"] += 1

    return {
        "player": name,
        "singles": {
            "elo": singles_ratings.get(name, 1000),
            "elo_history": singles_history.get(name, []),
            "matches": games,
            "wins": data["wins"],
            "losses": data["losses"],
            "win_pct": round(data["wins"] / games * 100, 1) if games else 0,
            "avg_points_won": round(data["points_won"] / games, 1) if games else 0,
            "avg_points_lost": round(data["points_lost"] / games, 1) if games else 0,
            "current_streak": current_streak,
            "longest_win_streak": longest_win,
            "longest_loss_streak": longest_loss,
            "head_to_head": [
                {"opponent": opp, "wins": v["wins"], "losses": v["losses"]}
                for opp, v in h2h.items()
            ],
        },
        "doubles": {
            "elo": doubles_ratings.get(name, 1000),
            "elo_history": doubles_history.get(name, []),
        },
    }
