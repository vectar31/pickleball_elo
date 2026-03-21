"""
One-time script to create player accounts in the DB.

Default password for all players: "pickleball"
Players are sourced from input.json.

Usage:
    python seed.py                        # use default password "pickleball"
    python seed.py --password mypassword  # use a custom default password
"""

import argparse
import json
import os
import sys

import database
from auth import hash_password

DEFAULT_PASSWORD = "pickleball"


def seed_players(default_password: str):
    database.init_db()

    # Load player list from input.json
    if not os.path.exists("input.json"):
        print("ERROR: input.json not found")
        sys.exit(1)

    with open("input.json") as f:
        data = json.load(f)

    players = data.get("players", [])
    if not players:
        print("No players found in input.json")
        return

    created = 0
    skipped = 0
    for name in players:
        if database.player_exists(name):
            print(f"  skip  {name} (already exists)")
            skipped += 1
        else:
            database.create_player(name, hash_password(default_password))
            print(f"  added {name}")
            created += 1

    print(f"\nDone. Created: {created}, Skipped: {skipped}")
    print(f"Default password: '{default_password}'")
    print("Share this password with your club members and ask them to note it down.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--password", default=DEFAULT_PASSWORD, help="Default password for all players")
    args = parser.parse_args()
    seed_players(args.password)
