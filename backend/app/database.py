import json
import os
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import sqlite3

DB_PATH = os.environ.get("DB_PATH", "data/pickleball.db")

# Single community for v1
COMMUNITY_ID = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
COMMUNITY_NAME = "Akhand Bharat Pickleball Club"
ADMIN_PLAYERS = {"Swapnil", "Piyush"}


def _ensure_dir():
    dir_name = os.path.dirname(DB_PATH)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)


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
            CREATE TABLE IF NOT EXISTS communities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL
            );

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
        _seed_community(conn)


def _add_col(conn, table, column, definition):
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def _migrate_schema(conn):
    """Add new columns to existing tables if missing. Safe to run multiple times."""
    # Players
    _add_col(conn, "players", "id", "TEXT")
    _add_col(conn, "players", "display_name", "TEXT")
    _add_col(conn, "players", "location", "TEXT")
    _add_col(conn, "players", "photo_url", "TEXT")
    _add_col(conn, "players", "created_at", "TEXT")

    # Admin role
    _add_col(conn, "players", "is_admin", "INTEGER NOT NULL DEFAULT 0")
    for admin in ADMIN_PLAYERS:
        conn.execute("UPDATE players SET is_admin=1 WHERE name=?", (admin,))

    # Generate UUIDs for players that don't have one yet
    rows = conn.execute("SELECT name FROM players WHERE id IS NULL").fetchall()
    for row in rows:
        conn.execute(
            "UPDATE players SET id=?, display_name=?, created_at=? WHERE name=?",
            (str(uuid.uuid4()), row["name"], datetime.now(timezone.utc).isoformat(), row["name"])
        )

    # Singles matches
    _add_col(conn, "singles_matches", "status", "TEXT NOT NULL DEFAULT 'accepted'")
    _add_col(conn, "singles_matches", "submitted_by", "TEXT NOT NULL DEFAULT ''")
    _add_col(conn, "singles_matches", "submitted_at", "TEXT NOT NULL DEFAULT ''")
    _add_col(conn, "singles_matches", "community_id", "TEXT")
    _add_col(conn, "singles_matches", "created_at", "TEXT NOT NULL DEFAULT ''")
    conn.execute("UPDATE singles_matches SET community_id=? WHERE community_id IS NULL", (COMMUNITY_ID,))

    # Doubles matches
    _add_col(conn, "doubles_matches", "status", "TEXT NOT NULL DEFAULT 'accepted'")
    _add_col(conn, "doubles_matches", "submitted_by", "TEXT NOT NULL DEFAULT ''")
    _add_col(conn, "doubles_matches", "submitted_at", "TEXT NOT NULL DEFAULT ''")
    _add_col(conn, "doubles_matches", "community_id", "TEXT")
    _add_col(conn, "doubles_matches", "created_at", "TEXT NOT NULL DEFAULT ''")
    conn.execute("UPDATE doubles_matches SET community_id=? WHERE community_id IS NULL", (COMMUNITY_ID,))


def _seed_community(conn):
    conn.execute(
        "INSERT OR IGNORE INTO communities (id, name) VALUES (?, ?)",
        (COMMUNITY_ID, COMMUNITY_NAME)
    )


def _load_json(filename):
    """Try to load a JSON file from cwd or parent directory."""
    for path in [filename, os.path.join("..", filename)]:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    return None


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


# ── Communities ───────────────────────────────────────────────────────────────

def get_community(community_id: str):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM communities WHERE id=?", (community_id,)).fetchone()
    return dict(row) if row else None


# ── Players ───────────────────────────────────────────────────────────────────

def _player_row_to_dict(row) -> dict:
    return {
        "id": row["id"],
        "name": row["name"],
        "display_name": row["display_name"] or row["name"],
        "location": row["location"],
        "photo_url": row["photo_url"],
        "created_at": row["created_at"],
    }


def get_all_players():
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM players ORDER BY name").fetchall()
    return [_player_row_to_dict(r) for r in rows]


def get_player(name: str):
    """Get raw player row by name (used for auth)."""
    with get_db() as conn:
        return conn.execute("SELECT * FROM players WHERE name=?", (name,)).fetchone()


def get_player_by_id(player_id: str):
    """Get player by UUID or name."""
    with get_db() as conn:
        row = conn.execute("SELECT * FROM players WHERE id=?", (player_id,)).fetchone()
        if not row:
            row = conn.execute("SELECT * FROM players WHERE name=?", (player_id,)).fetchone()
    return _player_row_to_dict(row) if row else None


def create_player(name: str, password_hash: str):
    player_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO players (name, password_hash, id, display_name, created_at) VALUES (?, ?, ?, ?, ?)",
            (name, password_hash, player_id, name, now)
        )


def player_exists(name: str) -> bool:
    with get_db() as conn:
        row = conn.execute("SELECT 1 FROM players WHERE name=?", (name,)).fetchone()
    return row is not None


def is_admin(name: str) -> bool:
    with get_db() as conn:
        row = conn.execute("SELECT is_admin FROM players WHERE name=?", (name,)).fetchone()
    return bool(row and row["is_admin"])


def update_player_profile(name: str, display_name: str = None, location: str = None, photo_url: str = None):
    with get_db() as conn:
        if display_name is not None:
            conn.execute("UPDATE players SET display_name=? WHERE name=?", (display_name, name))
        if location is not None:
            conn.execute("UPDATE players SET location=? WHERE name=?", (location, name))
        if photo_url is not None:
            conn.execute("UPDATE players SET photo_url=? WHERE name=?", (photo_url, name))


# ── Singles Matches ───────────────────────────────────────────────────────────

def get_singles_matches(status: str = None, player_name: str = None, limit: int = 20, cursor: int = None):
    with get_db() as conn:
        where, params = [], []
        if status:
            where.append("status=?")
            params.append(status)
        if player_name:
            where.append("(player1=? OR player2=?)")
            params.extend([player_name, player_name])
        if cursor:
            where.append("id < ?")
            params.append(cursor)
        sql = "SELECT * FROM singles_matches"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += f" ORDER BY id DESC LIMIT {int(limit) + 1}"
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def get_singles_match_by_id(match_id: int):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM singles_matches WHERE id=?", (match_id,)).fetchone()
    return dict(row) if row else None


def insert_singles_match(date: str, player1: str, player2: str, score1: int, score2: int, submitted_by: str):
    now = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO singles_matches (date, player1, player2, score1, score2, status, submitted_by, submitted_at, community_id, created_at) VALUES (?, ?, ?, ?, ?, 'pending_validation', ?, ?, ?, ?)",
            (date, player1, player2, score1, score2, submitted_by, now, COMMUNITY_ID, now)
        )


def update_singles_match_status(match_id: int, status: str):
    with get_db() as conn:
        conn.execute("UPDATE singles_matches SET status=? WHERE id=?", (status, match_id))


# ── Doubles Matches ───────────────────────────────────────────────────────────

def _row_to_doubles_match(r) -> dict:
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
        "community_id": r["community_id"],
        "created_at": r["created_at"],
    }


def get_doubles_matches(status: str = None, player_name: str = None, limit: int = 20, cursor: int = None):
    with get_db() as conn:
        where, params = [], []
        if status:
            where.append("status=?")
            params.append(status)
        if player_name:
            where.append("(team1_p1=? OR team1_p2=? OR team2_p1=? OR team2_p2=?)")
            params.extend([player_name, player_name, player_name, player_name])
        if cursor:
            where.append("id < ?")
            params.append(cursor)
        sql = "SELECT * FROM doubles_matches"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += f" ORDER BY id DESC LIMIT {int(limit) + 1}"
        rows = conn.execute(sql, params).fetchall()
    return [_row_to_doubles_match(r) for r in rows]


def get_doubles_match_by_id(match_id: int):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM doubles_matches WHERE id=?", (match_id,)).fetchone()
    return _row_to_doubles_match(row) if row else None


def insert_doubles_match(date: str, team1: list, team2: list, score1: int, score2: int, submitted_by: str):
    now = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO doubles_matches (date, team1_p1, team1_p2, team2_p1, team2_p2, score1, score2, status, submitted_by, submitted_at, community_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending_validation', ?, ?, ?, ?)",
            (date, team1[0], team1[1], team2[0], team2[1], score1, score2, submitted_by, now, COMMUNITY_ID, now)
        )


def update_doubles_match_status(match_id: int, status: str):
    with get_db() as conn:
        conn.execute("UPDATE doubles_matches SET status=? WHERE id=?", (status, match_id))


# ── Unified match lookup ──────────────────────────────────────────────────────

def get_any_match_by_id(match_id: int):
    """Search both tables. Returns (match_dict, format_str) or (None, None)."""
    m = get_singles_match_by_id(match_id)
    if m:
        return m, "singles"
    m = get_doubles_match_by_id(match_id)
    if m:
        return m, "doubles"
    return None, None


# ── Pending matches ───────────────────────────────────────────────────────────

def get_pending_for_player(player: str) -> list:
    """Return all matches pending this player's confirmation as a flat PendingMatch list."""
    with get_db() as conn:
        singles_rows = conn.execute(
            """SELECT * FROM singles_matches
               WHERE status='pending_validation' AND submitted_by != ?
               AND (player1=? OR player2=?)""",
            (player, player, player)
        ).fetchall()

        doubles_rows = conn.execute(
            """SELECT * FROM doubles_matches
               WHERE status='pending_validation' AND submitted_by != ?
               AND (team1_p1=? OR team1_p2=? OR team2_p1=? OR team2_p2=?)""",
            (player, player, player, player, player)
        ).fetchall()

    items = []

    for r in singles_rows:
        expires_at = _add_hours(r["submitted_at"], 24)
        items.append({
            "match_id": str(r["id"]),
            "format": "singles",
            "community_id": r["community_id"],
            "submitted_by_name": r["submitted_by"],
            "played_at": r["date"],
            "expires_at": expires_at,
            "score": {"team1": [r["score1"]], "team2": [r["score2"]]},
            "team1": [r["player1"]],
            "team2": [r["player2"]],
        })

    for r in doubles_rows:
        m = _row_to_doubles_match(r)
        submitter_team = m["team1"] if m["submitted_by"] in m["team1"] else m["team2"]
        player_team = m["team1"] if player in m["team1"] else m["team2"]
        if player_team == submitter_team:
            continue
        expires_at = _add_hours(m["submitted_at"], 24)
        items.append({
            "match_id": str(r["id"]),
            "format": "doubles",
            "community_id": m["community_id"],
            "submitted_by_name": m["submitted_by"],
            "played_at": m["date"],
            "expires_at": expires_at,
            "score": {"team1": [m["score1"]], "team2": [m["score2"]]},
            "team1": m["team1"],
            "team2": m["team2"],
        })

    return items


def _add_hours(iso_str: str, hours: int) -> str:
    try:
        return (datetime.fromisoformat(iso_str) + timedelta(hours=hours)).isoformat()
    except (ValueError, TypeError):
        return ""


# ── Auto-accept ───────────────────────────────────────────────────────────────

def auto_accept_expired(hours: int = 24):
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


# ── Feed ──────────────────────────────────────────────────────────────────────

def get_feed(limit: int = 20, cursor: str = None):
    """
    Return recent accepted matches from both tables sorted by date desc.
    Cursor format: '{date}_{format}_{id}'
    """
    cursor_date = cursor_format = None
    cursor_id = None
    if cursor:
        try:
            parts = cursor.rsplit("_", 2)
            cursor_date, cursor_format, cursor_id = parts[0], parts[1], int(parts[2])
        except (ValueError, IndexError):
            pass

    with get_db() as conn:
        def build_cursor_clause(fmt):
            if cursor_date is None:
                return "", []
            # If same format, use exact id; otherwise just date boundary
            if fmt == cursor_format:
                return " AND (date < ? OR (date = ? AND id < ?))", [cursor_date, cursor_date, cursor_id]
            else:
                return " AND date <= ?", [cursor_date]

        s_clause, s_params = build_cursor_clause("singles")
        singles = conn.execute(
            f"SELECT id, date, player1, player2, score1, score2, community_id, submitted_by FROM singles_matches WHERE status='accepted'{s_clause} ORDER BY date DESC, id DESC LIMIT ?",
            s_params + [limit]
        ).fetchall()

        d_clause, d_params = build_cursor_clause("doubles")
        doubles = conn.execute(
            f"SELECT id, date, team1_p1, team1_p2, team2_p1, team2_p2, score1, score2, community_id, submitted_by FROM doubles_matches WHERE status='accepted'{d_clause} ORDER BY date DESC, id DESC LIMIT ?",
            d_params + [limit]
        ).fetchall()

    items = []
    for r in singles:
        items.append({
            "id": r["id"], "date": r["date"], "format": "singles",
            "team1": [r["player1"]], "team2": [r["player2"]],
            "score": {"team1": [r["score1"]], "team2": [r["score2"]]},
            "community_id": r["community_id"], "submitted_by": r["submitted_by"],
        })
    for r in doubles:
        items.append({
            "id": r["id"], "date": r["date"], "format": "doubles",
            "team1": [r["team1_p1"], r["team1_p2"]], "team2": [r["team2_p1"], r["team2_p2"]],
            "score": {"team1": [r["score1"]], "team2": [r["score2"]]},
            "community_id": r["community_id"], "submitted_by": r["submitted_by"],
        })

    items.sort(key=lambda x: (x["date"], x["id"]), reverse=True)
    items = items[:limit]

    next_cursor = None
    if len(items) == limit:
        last = items[-1]
        next_cursor = f"{last['date']}_{last['format']}_{last['id']}"

    return items, next_cursor
