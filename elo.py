import json
import os
from collections import defaultdict
from datetime import datetime
from math import pow

DEFAULT_RATING = 1000
K = 32

def expected_score(ra, rb):
    return 1 / (1 + pow(10, (rb - ra) / 400))

def update_elo(ra, rb, result_a):
    ea = expected_score(ra, rb)
    ra_new = ra + K * (result_a - ea)
    rb_new = rb + K * ((1 - result_a) - (1 - ea))
    return ra_new, rb_new

def today():
    return datetime.today().strftime("%Y-%m-%d")

# File paths
INPUT_FILE = "input.json"
LIVE_MATCHES_FILE = "live_matches.json"
OUTPUT_RATINGS = "output/ratings.json"
OUTPUT_HISTORY = "output/history.json"

def load_matches():
    input_matches = _load_json(INPUT_FILE).get("matches", [])
    live_matches = _load_json(LIVE_MATCHES_FILE)
    return input_matches + live_matches

def load_players():
    input_data = _load_json(INPUT_FILE)
    return set(input_data.get("players", []))

def _load_json(path):
    if not os.path.exists(path):
        return [] if path.endswith(".json") else {}
    with open(path, 'r') as f:
        return json.load(f)

def _save_json(path, data):
    dir_name = os.path.dirname(path)
    if dir_name:  # Only make directory if there's one to make
        os.makedirs(dir_name, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def compute_ratings_and_history():
    matches = load_matches()
    players = list(load_players())
    ratings = {p: DEFAULT_RATING for p in players}
    history = {p: [(0, DEFAULT_RATING)] for p in players}

    match_number = 1

    for match in matches:
        p1, p2 = match["player1"], match["player2"]
        s1, s2 = match["score1"], match["score2"]

        for p in [p1, p2]:
            if p not in ratings:
                ratings[p] = DEFAULT_RATING
                history[p] = [(0, DEFAULT_RATING)]

        if s1 == s2:
            continue  # skip ties

        winner, loser = (p1, p2) if s1 > s2 else (p2, p1)
        rw, rl = ratings[winner], ratings[loser]
        rw_new, rl_new = update_elo(rw, rl, 1)

        ratings[winner] = round(rw_new, 2)
        ratings[loser] = round(rl_new, 2)
        history[winner].append((match_number, round(rw_new, 2)))
        history[loser].append((match_number, round(rl_new, 2)))

        match_number += 1

    _save_json(OUTPUT_RATINGS, ratings)
    _save_json(OUTPUT_HISTORY, history)

    return ratings, history, matches


def add_match(player1, player2, score1, score2):
    if player1 == player2:
        return "❌ Players must be different."
    if score1 == score2:
        return "❌ No ties allowed."

    match = {
        "date": today(),
        "player1": player1,
        "player2": player2,
        "score1": score1,
        "score2": score2
    }

    existing = _load_json(LIVE_MATCHES_FILE)
    existing.append(match)
    _save_json(LIVE_MATCHES_FILE, existing)

    return "✅ Match added!"
