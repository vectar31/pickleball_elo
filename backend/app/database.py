import sqlite3
import json
import os
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

DB_PATH = os.environ.get("DB_PATH", "data/pickleball.db")


def _ensure_dir():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


def get_connection():
    _ensure_dir()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def get_db():
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS players (
                name TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS singles_matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                player1 TEXT NOT NULL,
                player2 TEXT NOT NULL,
                score1 INTEGER NOT NULL,
                score2 INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS doubles_matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                team1_p1 TEXT NOT NULL,
                team1_p2 TEXT NOT NULL,
                team2_p1 TEXT NOT NULL,
                team2_p2 TEXT NOT NULL,
                score1 INTEGER NOT NULL,
                score2 INTEGER NOT NULL
            );
        """)
        _migrate_from_json(conn)
        _migrate_schema(conn)


def _add_col(conn, table, column, definition):
    """Add a column only if it doesn't already exist."""
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _migrate_schema(conn):
    """Add validation columns to existing tables if missing. Existing rows default to accepted."""
    for table in ("singles_matches", "doubles_matches"):
        _add_col(conn, table, "status", "TEXT NOT NULL DEFAULT 'accepted'")
        _add_col(conn, table, "submitted_by", "TEXT NOT NULL DEFAULT ''")
        _add_col(conn, table, "submitted_at", "TEXT NOT NULL DEFAULT ''")


def _load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _migrate_from_json(conn):
    """One-time migration from JSON files to SQLite. Skips if data already exists."""
    singles_count = conn.execute("SELECT COUNT(*) FROM singles_matches").fetchone()[0]
    doubles_count = conn.execute("SELECT COUNT(*) FROM doubles_matches").fetchone()[0]

    if singles_count == 0:
        input_data = _load_json("input.json") or {}
        live_matches = _load_json("live_matches.json") or []
        all_singles = input_data.get("matches", []) + (live_matches if isinstance(live_matches, list) else [])
        for m in all_singles:
            conn.execute(
                "INSERT INTO singles_matches (date, player1, player2, score1, score2) VALUES (?, ?, ?, ?, ?)",
                (m["date"], m["player1"], m["player2"], m["score1"], m["score2"])
            )
        print(f"[db] Migrated {len(all_singles)} singles matches from JSON")

    if doubles_count == 0:
        doubles = _load_json("doubles_results.json") or []
        for m in doubles:
            t1, t2 = m["team1"], m["team2"]
            conn.execute(
                "INSERT INTO doubles_matches (date, team1_p1, team1_p2, team2_p1, team2_p2, score1, score2) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (m["date"], t1[0], t1[1], t2[0], t2[1], m["score1"], m["score2"])
            )
        print(f"[db] Migrated {len(doubles)} doubles matches from JSON")


# ── Players ───────────────────────────────────────────────────────────────────

def get_all_players():
    with get_db() as conn:
        rows = conn.execute("SELECT name FROM players ORDER BY name").fetchall()
    return [r["name"] for r in rows]


def get_player(name: str):
    with get_db() as conn:
        return conn.execute("SELECT * FROM players WHERE name = ?", (name,)).fetchone()


def create_player(name: str, password_hash: str):
    with get_db() as conn:
        conn.execute("INSERT INTO players (name, password_hash) VALUES (?, ?)", (name, password_hash))


def player_exists(name: str) -> bool:
    with get_db() as conn:
        row = conn.execute("SELECT 1 FROM players WHERE name = ?", (name,)).fetchone()
    return row is not None


# ── Singles Matches ───────────────────────────────────────────────────────────

def get_singles_matches(status: str = None):
    """Return all singles matches, optionally filtered by status."""
    with get_db() as conn:
        if status:
            rows = conn.execute(
                "SELECT * FROM singles_matches WHERE status=? ORDER BY id ASC", (status,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM singles_matches ORDER BY id ASC").fetchall()
    return [dict(r) for r in rows]


def get_singles_match_by_id(match_id: int):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM singles_matches WHERE id=?", (match_id,)).fetchone()
    return dict(row) if row else None


def insert_singles_match(date: str, player1: str, player2: str, score1: int, score2: int, submitted_by: str):
    submitted_at = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO singles_matches (date, player1, player2, score1, score2, status, submitted_by, submitted_at) VALUES (?, ?, ?, ?, ?, 'pending_validation', ?, ?)",
            (date, player1, player2, score1, score2, submitted_by, submitted_at)
        )


def update_singles_match_status(match_id: int, status: str):
    with get_db() as conn:
        conn.execute("UPDATE singles_matches SET status=? WHERE id=?", (status, match_id))


# ── Doubles Matches ───────────────────────────────────────────────────────────

def _row_to_doubles_match(r):
    return {
        "id": r["id"],
        "date": r["date"],
        "team1": [r["team1_p1"], r["team1_p2"]],
        "team2": [r["team2_p1"], r["team2_p2"]],
        "score1": r["score1"],
        "score2": r["score2"],
        "status": r["status"],
        "submitted_by": r["submitted_by"],
        "submitted_at": r["submitted_at"],
    }


def get_doubles_matches(status: str = None):
    """Return all doubles matches, optionally filtered by status."""
    with get_db() as conn:
        if status:
            rows = conn.execute(
                "SELECT * FROM doubles_matches WHERE status=? ORDER BY id ASC", (status,)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM doubles_matches ORDER BY id ASC").fetchall()
    return [_row_to_doubles_match(r) for r in rows]


def get_doubles_match_by_id(match_id: int):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM doubles_matches WHERE id=?", (match_id,)).fetchone()
    return _row_to_doubles_match(row) if row else None


def insert_doubles_match(date: str, team1: list, team2: list, score1: int, score2: int, submitted_by: str):
    submitted_at = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO doubles_matches (date, team1_p1, team1_p2, team2_p1, team2_p2, score1, score2, status, submitted_by, submitted_at) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending_validation', ?, ?)",
            (date, team1[0], team1[1], team2[0], team2[1], score1, score2, submitted_by, submitted_at)
        )


def update_doubles_match_status(match_id: int, status: str):
    with get_db() as conn:
        conn.execute("UPDATE doubles_matches SET status=? WHERE id=?", (status, match_id))


# ── Validation helpers ────────────────────────────────────────────────────────

def get_pending_for_player(player: str):
    """Return all matches pending this player's confirmation (they are the opponent, not the submitter)."""
    with get_db() as conn:
        singles_rows = conn.execute(
            """SELECT * FROM singles_matches
               WHERE status='pending_validation'
               AND submitted_by != ?
               AND (player1=? OR player2=?)""",
            (player, player, player)
        ).fetchall()

        doubles_rows = conn.execute(
            """SELECT * FROM doubles_matches
               WHERE status='pending_validation'
               AND submitted_by != ?
               AND (team1_p1=? OR team1_p2=? OR team2_p1=? OR team2_p2=?)""",
            (player, player, player, player, player)
        ).fetchall()

    singles = [dict(r) for r in singles_rows]

    doubles = []
    for r in doubles_rows:
        m = _row_to_doubles_match(r)
        # Only include if player is on the opposing team from the submitter
        submitter_team = m["team1"] if m["submitted_by"] in m["team1"] else m["team2"]
        player_team = m["team1"] if player in m["team1"] else m["team2"]
        if player_team != submitter_team:
            doubles.append(m)

    return {"singles": singles, "doubles": doubles}


def auto_accept_expired(hours: int = 24):
    """Auto-accept matches that have been pending longer than `hours` hours."""
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    with get_db() as conn:
        conn.execute(
            "UPDATE singles_matches SET status='accepted' WHERE status='pending_validation' AND submitted_at != '' AND submitted_at < ?",
            (cutoff,)
        )
        conn.execute(
            "UPDATE doubles_matches SET status='accepted' WHERE status='pending_validation' AND submitted_at != '' AND submitted_at < ?",
            (cutoff,)
        )
