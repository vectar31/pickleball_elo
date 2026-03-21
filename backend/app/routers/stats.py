from collections import defaultdict

from fastapi import APIRouter, HTTPException

from app import database
from app.elo.engine import compute_ratings_from_data, compute_doubles_ratings_from_data

router = APIRouter(prefix="/stats", tags=["stats"])


def _accepted(matches):
    return [m for m in matches if m.get("status", "accepted") == "accepted"]


def _compute_singles_stats(matches):
    stats = defaultdict(lambda: {
        "wins": 0, "losses": 0, "points_won": 0, "points_lost": 0, "history": []
    })
    for m in matches:
        p1, p2, s1, s2 = m["player1"], m["player2"], m["score1"], m["score2"]
        winner = p1 if s1 > s2 else p2
        loser = p2 if s1 > s2 else p1
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
    count = sum(1 for _ in (r for r in reversed(history) if r == last) )
    # recalculate properly
    count = 0
    for r in reversed(history):
        if r == last:
            count += 1
        else:
            break

    def max_streak(seq, target):
        best = cur = 0
        for r in seq:
            cur = cur + 1 if r == target else 0
            best = max(best, cur)
        return best

    return f"{count}{last}", max_streak(history, "W"), max_streak(history, "L")


@router.get("/singles")
def get_singles_stats():
    database.auto_accept_expired(hours=24)
    accepted = _accepted(database.get_singles_matches(limit=100000, order="asc"))
    players = [p["name"] for p in database.get_all_players()]
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


@router.get("/doubles")
def get_doubles_stats():
    database.auto_accept_expired(hours=24)
    accepted = _accepted(database.get_doubles_matches(limit=100000, order="asc"))
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


@router.get("/player/{name}")
def get_player_stats(name: str):
    if not database.player_exists(name):
        raise HTTPException(status_code=404, detail="Player not found")

    database.auto_accept_expired(hours=24)
    accepted_singles = _accepted(database.get_singles_matches())
    accepted_doubles = _accepted(database.get_doubles_matches())
    players = [p["name"] for p in database.get_all_players()]

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
            h2h[opp]["wins" if won else "losses"] += 1

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
