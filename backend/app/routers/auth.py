from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from app import database
from app.auth.jwt import verify_password, create_access_token

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login")
def login(form: OAuth2PasswordRequestForm = Depends()):
    row = database.get_player(form.username)
    if not row or not verify_password(form.password, row["password_hash"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    token = create_access_token(form.username)
    return {"access_token": token, "token_type": "bearer"}
