"""
Score prediction model.

Two regressors (home_score, away_score) trained on the same team-form features as
the win-probability model. Backtested against two baselines:
  1. Market-implied score, derived algebraically from the closing spread + total --
     this is what a sportsbook's own numbers already imply, no model needed.
  2. A truly naive baseline: each team's own rolling scoring average.

If our model can't beat baseline #1, that's the same "market already knows" pattern
we've seen on win probability, props, QB changes, and weather. Worth checking honestly
rather than assuming either way.
"""
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_parquet("matchup_dataset.parquet")

FEATURES = [
    "home_avg_pts_for_L5", "home_avg_pts_against_L5", "home_avg_margin_L5",
    "home_win_pct_L5", "home_season_win_pct", "home_rest_days",
    "home_qb_changed", "home_qb_starts_with_team",
    "away_avg_pts_for_L5", "away_avg_pts_against_L5", "away_avg_margin_L5",
    "away_win_pct_L5", "away_season_win_pct", "away_rest_days",
    "away_qb_changed", "away_qb_starts_with_team",
    "is_dome", "temp", "wind",
    "spread_line", "total_line",
]

data = df.dropna(subset=FEATURES + ["home_score", "away_score"]).copy()
data = data[data["season"] >= 2005].reset_index(drop=True)

train = data[data["season"] <= 2023]
test = data[data["season"] == 2024].copy()

X_train = train[FEATURES]
X_test = test[FEATURES]

models = {}
for target in ["home_score", "away_score"]:
    model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8)
    model.fit(X_train, train[target])
    models[target] = model
    test[f"pred_{target}"] = model.predict(X_test)

# Market-implied score, algebraically derived from spread + total:
# total_line = home_score + away_score (expected)
# spread_line = home_score - away_score (expected; positive = home favored, per our convention)
test["market_implied_home"] = (test["total_line"] + test["spread_line"]) / 2
test["market_implied_away"] = (test["total_line"] - test["spread_line"]) / 2

print("=== SCORE PREDICTION MAE (2024 holdout season) ===")
for side in ["home", "away"]:
    model_mae = mean_absolute_error(test[f"{side}_score"], test[f"pred_{side}_score"])
    market_mae = mean_absolute_error(test[f"{side}_score"], test[f"market_implied_{side}"])
    naive_mae = mean_absolute_error(test[f"{side}_score"], test[f"{side}_avg_pts_for_L5"])
    print(f"{side.upper()} score -- model: {model_mae:.2f} pts | "
          f"market-implied: {market_mae:.2f} pts | naive (own L5 avg): {naive_mae:.2f} pts")

for target in models:
    models[target].save_model(f"{target}_model.json")

print("\nModels saved: home_score_model.json, away_score_model.json")
