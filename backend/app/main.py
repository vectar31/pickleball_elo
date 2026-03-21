from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import database
from app.routers import auth, players, matches, ratings, stats, communities, feed

app = FastAPI(title="Pickleball ELO API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_STATUS_TITLES = {
    400: "Bad Request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not Found",
    409: "Conflict",
    422: "Unprocessable Entity",
    500: "Internal Server Error",
}


@app.exception_handler(HTTPException)
async def problem_json_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "type": f"https://httpstatuses.com/{exc.status_code}",
            "title": _STATUS_TITLES.get(exc.status_code, "Error"),
            "status": exc.status_code,
            "detail": exc.detail,
        },
        media_type="application/problem+json",
    )


@app.on_event("startup")
def startup():
    database.init_db()


app.include_router(auth.router)
app.include_router(players.router)
app.include_router(matches.router)
app.include_router(ratings.router)
app.include_router(stats.router)
app.include_router(communities.router)
app.include_router(feed.router)
