from fastapi import APIRouter

from app import database

router = APIRouter(prefix="/players", tags=["players"])


@router.get("")
def get_players():
    return {"players": database.get_all_players()}
