"""
Unified weekly prediction pipeline.

Run this once a week (e.g. Tuesday, after previous week's games are final and this
week's schedule/injury news is out) to regenerate predictions for both:
  1. Team-level spread/win predictions (learning project -- does NOT beat the market,
     see backtest in train_and_backtest.py)
  2. Player prop directional predictions (the more realistic place for any real edge,
     still unverified against real book lines)

Output: a single JSON the Flutter app / FastAPI backend can serve to the user.
"""
import json
import pandas as pd
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
import nfl_data_py as nfl

from build_dataset import (
    load_games, build_team_game_log, add_rolling_form,
    add_qb_continuity_features, add_weather_features,
)

SCORE_FEATURES = [
    "home_avg_pts_for_L5", "home_avg_pts_against_L5", "home_avg_margin_L5",
    "home_win_pct_L5", "home_season_win_pct", "home_rest_days",
    "home_qb_changed", "home_qb_starts_with_team",
    "away_avg_pts_for_L5", "away_avg_pts_against_L5", "away_avg_margin_L5",
    "away_win_pct_L5", "away_season_win_pct", "away_rest_days",
    "away_qb_changed", "away_qb_starts_with_team",
    "is_dome", "temp", "wind",
    "spread_line", "total_line",
]


def get_upcoming_team_predictions(season: int, week: int):
    games = load_games()
    games = add_weather_features(games)  # adds is_dome, fills temp/wind for domes
    long_df = build_team_game_log(games)
    long_df = add_rolling_form(long_df, window=5)
    long_df = add_qb_continuity_features(long_df)

    model = XGBClassifier()
    model.load_model("model.json")

    feature_cols = [
        "home_avg_pts_for_L5", "home_avg_pts_against_L5", "home_avg_margin_L5",
        "home_win_pct_L5", "home_season_win_pct", "home_rest_days",
        "home_qb_changed", "home_qb_starts_with_team",
        "away_avg_pts_for_L5", "away_avg_pts_against_L5", "away_avg_margin_L5",
        "away_win_pct_L5", "away_season_win_pct", "away_rest_days",
        "away_qb_changed", "away_qb_starts_with_team",
        "is_dome", "temp", "wind",
        "spread_line",
    ]

    upcoming = games[(games["season"] == season) & (games["week"] == week)].copy()
    if upcoming.empty:
        return []
    # raw games.csv uses home_rest / away_rest; training used home_rest_days / away_rest_days
    upcoming = upcoming.rename(columns={"home_rest": "home_rest_days", "away_rest": "away_rest_days"})

    feat_source = long_df[["game_id", "team", "is_home",
                            "avg_pts_for_L5", "avg_pts_against_L5", "avg_margin_L5",
                            "win_pct_L5", "season_win_pct", "qb_changed", "qb_starts_with_team"]]

    home_feats = feat_source[feat_source["is_home"] == 1].rename(
        columns={c: f"home_{c}" for c in feat_source.columns if c not in ["game_id", "team", "is_home"]})
    away_feats = feat_source[feat_source["is_home"] == 0].rename(
        columns={c: f"away_{c}" for c in feat_source.columns if c not in ["game_id", "team", "is_home"]})

    upcoming = upcoming.merge(home_feats.drop(columns=["team", "is_home"]), on="game_id", how="left")
    upcoming = upcoming.merge(away_feats.drop(columns=["team", "is_home"]), on="game_id", how="left")

    # Week 1 games have no season_win_pct yet (no games played this season) -- fill with a
    # neutral 0.5 prior rather than dropping the row. All other features (L5 form, rest)
    # carry over from the end of the prior season and are already present.
    for col in ["home_season_win_pct", "away_season_win_pct"]:
        upcoming[col] = upcoming[col].fillna(0.5)
    # qb_changed is NaN only for a team's very first game in the whole dataset (1999) --
    # essentially never hit for a live upcoming week, but fill defensively anyway.
    for col in ["home_qb_changed", "away_qb_changed"]:
        upcoming[col] = upcoming[col].fillna(0.0)

    # Outdoor games that haven't been played yet have no recorded temp/wind (that's
    # only known after the fact). Try a live forecast for each; if it's unavailable
    # (no internet, game too far out, stadium not in the coords table), fall back to
    # a neutral league-average value rather than dropping the game entirely.
    upcoming["weather_source"] = upcoming["is_dome"].map({1.0: "dome", 0.0: "recorded"})
    outdoor_missing = upcoming[(upcoming["is_dome"] == 0) & (upcoming["temp"].isna())]
    for idx, row in outdoor_missing.iterrows():
        forecast = None
        try:
            from live_weather import get_live_weather_forecast
            kickoff_iso = f"{row['gameday']}T13:00:00"  # approx kickoff if exact time unknown
            forecast = get_live_weather_forecast(row["home_team"], kickoff_iso)
        except Exception:
            forecast = None
        if forecast:
            upcoming.at[idx, "temp"] = forecast["temperature_f"]
            wind_num = "".join(c for c in forecast["wind_speed"] if c.isdigit())
            upcoming.at[idx, "wind"] = float(wind_num) if wind_num else 5.0
            upcoming.at[idx, "weather_source"] = "live_forecast"
        else:
            upcoming.at[idx, "temp"] = 60.0  # neutral fallback, not a real forecast
            upcoming.at[idx, "wind"] = 5.0
            upcoming.at[idx, "weather_source"] = "fallback_no_forecast"

    upcoming = upcoming.dropna(subset=feature_cols)
    if upcoming.empty:
        return []

    upcoming["model_home_win_prob"] = model.predict_proba(upcoming[feature_cols])[:, 1]

    # Score prediction, if the score models are available (run score_model.py first)
    predicted_scores = {}
    try:
        home_score_model = XGBRegressor()
        home_score_model.load_model("home_score_model.json")
        away_score_model = XGBRegressor()
        away_score_model.load_model("away_score_model.json")

        score_ready = upcoming.dropna(subset=SCORE_FEATURES)
        if not score_ready.empty:
            pred_home = home_score_model.predict(score_ready[SCORE_FEATURES])
            pred_away = away_score_model.predict(score_ready[SCORE_FEATURES])
            for gid, ph, pa in zip(score_ready["game_id"], pred_home, pred_away):
                rh, ra = round(float(ph)), round(float(pa))
                # NFL games (almost) never end in ties -- if rounding both raw
                # predictions to the nearest integer produces a tie, break it using
                # the direction of the *unrounded* prediction (whichever was actually
                # higher before rounding), nudging the winner up by 1 point.
                if rh == ra:
                    if ph >= pa:
                        rh += 1
                    else:
                        ra += 1
                predicted_scores[gid] = (rh, ra)
    except Exception as e:
        print(f"WARNING: score models not loaded ({e}). "
              f"Run `python score_model.py` to enable score predictions.")

    results = []
    for _, row in upcoming.iterrows():
        pred_score = predicted_scores.get(row["game_id"])
        entry = {
            "matchup": f"{row['away_team']} @ {row['home_team']}",
            "spread_line": row["spread_line"],
            "model_home_win_prob": round(float(row["model_home_win_prob"]), 3),
            "weather_source": row.get("weather_source", "unknown"),
            "note": "Model tends to AGREE with the market here, not beat it -- see backtest.",
        }
        if pred_score:
            entry["predicted_away_score"] = pred_score[1]
            entry["predicted_home_score"] = pred_score[0]

        # If this game has already been played, surface the real result too --
        # otherwise the dashboard always looks like it's "predicting" a finished game.
        if pd.notna(row.get("home_score")) and pd.notna(row.get("away_score")):
            actual_home_win = bool(row["home_score"] > row["away_score"])
            model_picked_home = row["model_home_win_prob"] > 0.5
            entry["final"] = True
            entry["actual_home_score"] = int(row["home_score"])
            entry["actual_away_score"] = int(row["away_score"])
            entry["model_correct"] = (model_picked_home == actual_home_win)
        else:
            entry["final"] = False

        results.append(entry)
    return results


def get_top_prop_candidates(season: int, week: int, stat_col: str = "receiving_yards", top_n: int = 10):
    weekly = nfl.import_weekly_data([season - 1, season])  # prior + current season for rolling history
    weekly = weekly[weekly["position"].isin(["WR", "TE"])].sort_values(["player_id", "season", "week"])

    grp = weekly.groupby("player_id")
    weekly[f"{stat_col}_L5_avg"] = grp[stat_col].transform(
        lambda s: s.shift(1).rolling(5, min_periods=2).mean())

    latest = weekly.sort_values(["season", "week"]).groupby("player_id").tail(1).copy()
    latest = latest.dropna(subset=[f"{stat_col}_L5_avg"])
    latest["projected"] = latest[f"{stat_col}_L5_avg"]  # placeholder projection surface
    top = latest.sort_values("projected", ascending=False).head(top_n)

    return top[["player_display_name", "recent_team", "projected"]].to_dict(orient="records")


def grade_past_predictions(log_path: str = "predictions_log.json"):
    """
    Compare every logged prediction against the actual final score (once available)
    and tag each as a hit or miss. Run this periodically (e.g. each Tuesday before
    generating the next week's predictions) to build a real accuracy record over
    the season instead of eyeballing individual results.
    """
    import os
    if not os.path.exists(log_path):
        return {"graded": 0, "message": "No prediction log yet."}

    with open(log_path) as f:
        log = json.load(f)

    games = load_games()
    graded_count = 0

    for entry in log:
        if entry.get("actual_home_win") is not None:
            continue  # already graded, don't re-check finished entries every run

        match = games[
            (games["season"] == entry["season"]) &
            (games["week"] == entry["week"]) &
            (games["home_team"] == entry["home_team"]) &
            (games["away_team"] == entry["away_team"])
        ]
        if match.empty or pd.isna(match.iloc[0]["home_score"]):
            continue  # game hasn't been played yet, nothing to grade

        row = match.iloc[0]
        actual_home_win = bool(row["home_score"] > row["away_score"])
        model_picked_home = entry["model_home_win_prob"] > 0.5

        entry["actual_home_win"] = actual_home_win
        entry["actual_score"] = f"{row['away_team']} {int(row['away_score'])} - {int(row['home_score'])} {row['home_team']}"
        entry["correct"] = (model_picked_home == actual_home_win)
        graded_count += 1

    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    graded_entries = [e for e in log if e.get("correct") is not None]
    accuracy = (sum(e["correct"] for e in graded_entries) / len(graded_entries)
                if graded_entries else None)

    return {
        "newly_graded": graded_count,
        "total_graded": len(graded_entries),
        "season_accuracy": round(accuracy, 3) if accuracy is not None else None,
    }


def log_predictions(team_preds, season, week, log_path: str = "predictions_log.json"):
    """Append this week's predictions to a running log, tagged with season/week/teams
    so grade_past_predictions can match them to results later."""
    import os
    log = []
    if os.path.exists(log_path):
        with open(log_path) as f:
            log = json.load(f)

    existing_keys = {(e["season"], e["week"], e["home_team"], e["away_team"]) for e in log}

    for pred in team_preds:
        away, home = pred["matchup"].split(" @ ")
        key = (season, week, home, away)
        if key in existing_keys:
            continue  # don't duplicate if this week was already logged before
        log.append({
            "season": season, "week": week,
            "home_team": home, "away_team": away,
            "spread_line": pred["spread_line"],
            "model_home_win_prob": pred["model_home_win_prob"],
            "actual_home_win": None,   # filled in later by grade_past_predictions
            "actual_score": None,
            "correct": None,
        })

    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)


if __name__ == "__main__":
    SEASON, WEEK = 2026, 1
    team_preds = get_upcoming_team_predictions(SEASON, WEEK)
    print(f"Team predictions for {SEASON} Week {WEEK}: {len(team_preds)} games found")
    for g in team_preds[:5]:
        print(" ", g)

    log_predictions(team_preds, SEASON, WEEK)
    grade_result = grade_past_predictions()
    print(f"\nGrading pass: {grade_result}")

    output = {"season": SEASON, "week": WEEK, "team_predictions": team_preds}
    with open("weekly_output.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nSaved weekly_output.json")