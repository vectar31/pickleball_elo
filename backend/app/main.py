from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import database
from app.routers import auth, players, matches, ratings, stats

app = FastAPI(title="Pickleball ELO API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    database.init_db()


app.include_router(auth.router)
app.include_router(players.router)
app.include_router(matches.router)
app.include_router(ratings.router)
app.include_router(stats.router)
