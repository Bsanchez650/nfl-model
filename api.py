"""
FastAPI backend for the NFL predictions app.

Endpoints:
  GET /predictions/{season}/{week}  -> team spread predictions for that week
  GET /health                       -> basic check

Run with: uvicorn api:app --reload --port 8000
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from functools import lru_cache

from weekly_pipeline import get_upcoming_team_predictions, log_predictions, grade_past_predictions

app = FastAPI(title="NFL Predictions API")

# Flutter app (mobile/web) will call this from a different origin -- allow it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/predictions/{season}/{week}")
def predictions(season: int, week: int):
    try:
        games = get_upcoming_team_predictions(season, week)
        log_predictions(games, season, week)  # so this week gets tracked automatically
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not games:
        raise HTTPException(status_code=404, detail="No games found or lines not posted yet for this week.")

    return {
        "season": season,
        "week": week,
        "games": games,
        "disclaimer": (
            "This model is trained on public team stats and, in backtesting, does not "
            "reliably beat the sportsbook closing line. It's a decision-support tool, "
            "not financial advice, and disagreeing with the market has not been shown "
            "to be more accurate than the market itself."
        ),
    }


@app.get("/track_record")
def track_record():
    """Grades any newly-finished games and returns your running accuracy so far."""
    try:
        result = grade_past_predictions()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return result