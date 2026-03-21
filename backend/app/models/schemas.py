from datetime import datetime
from typing import List

from pydantic import BaseModel, field_validator


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
