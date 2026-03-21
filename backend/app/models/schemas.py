from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, field_validator


# ── Input schemas ─────────────────────────────────────────────────────────────

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


class DisputeInput(BaseModel):
    reason: Optional[str] = None


# ── Response schemas (for OpenAPI docs) ───────────────────────────────────────

class PlayerResponse(BaseModel):
    id: str
    name: str
    display_name: str
    location: Optional[str]
    photo_url: Optional[str]
    created_at: Optional[str]


class EloEntry(BaseModel):
    player_id: str
    player_name: str
    photo_url: Optional[str]
    elo: float
    delta: float
    rank: int
    wins: int
    losses: int


class PendingMatch(BaseModel):
    match_id: str
    format: str
    community_id: Optional[str]
    submitted_by_name: str
    played_at: str
    expires_at: str
    score: dict
    team1: List[str]
    team2: List[str]
