from collections import defaultdict
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


def compute_ratings_from_data(matches: list, players: list):
    """Compute singles ELO ratings and history from a list of accepted matches."""
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
            continue

        winner, loser = (p1, p2) if s1 > s2 else (p2, p1)
        rw, rl = ratings[winner], ratings[loser]
        rw_new, rl_new = update_elo(rw, rl, 1)

        ratings[winner] = round(rw_new, 2)
        ratings[loser] = round(rl_new, 2)
        history[winner].append((match_number, round(rw_new, 2)))
        history[loser].append((match_number, round(rl_new, 2)))
        match_number += 1

    return ratings, history


def compute_doubles_ratings_from_data(matches: list):
    """Compute doubles ELO ratings and history from a list of accepted matches."""
    ratings = defaultdict(lambda: DEFAULT_RATING)
    history = defaultdict(lambda: [(0, DEFAULT_RATING)])
    match_number = 1

    for match in matches:
        team1 = match["team1"]
        team2 = match["team2"]
        s1, s2 = match["score1"], match["score2"]

        if s1 == s2 or set(team1) & set(team2):
            continue

        r1 = sum(ratings[p] for p in team1) / 2
        r2 = sum(ratings[p] for p in team2) / 2
        result = 1 if s1 > s2 else 0
        r1_new, r2_new = update_elo(r1, r2, result)

        for p in team1:
            ratings[p] = round(ratings[p] + (r1_new - r1), 2)
            history[p].append((match_number, ratings[p]))

        for p in team2:
            ratings[p] = round(ratings[p] + (r2_new - r2), 2)
            history[p].append((match_number, ratings[p]))

        match_number += 1

    return dict(ratings), dict(history)
