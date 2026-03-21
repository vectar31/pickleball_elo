"""
One-time script to create player accounts in the DB.
Default password for all players: "pickleball"

Usage:
    python seed.py                        # default password
    python seed.py --password mypassword  # custom password
"""

import argparse
import json
import os
import sys

from app import database
from app.auth.jwt import hash_password

DEFAULT_PASSWORD = "pickleball"


def seed_players(default_password: str):
    database.init_db()

    input_path = os.path.join(os.path.dirname(__file__), "..", "input.json")
    if not os.path.exists(input_path):
        print("ERROR: input.json not found")
        sys.exit(1)

    with open(input_path) as f:
        data = json.load(f)

    players = data.get("players", [])
    if not players:
        print("No players found in input.json")
        return

    created = skipped = 0
    for name in players:
        if database.player_exists(name):
            print(f"  skip  {name}")
            skipped += 1
        else:
            database.create_player(name, hash_password(default_password))
            print(f"  added {name}")
            created += 1

    print(f"\nDone. Created: {created}, Skipped: {skipped}")
    print(f"Default password: '{default_password}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--password", default=DEFAULT_PASSWORD)
    args = parser.parse_args()
    seed_players(args.password)
