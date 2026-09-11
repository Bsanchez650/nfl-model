"""
Train a model to predict home team win probability, then check two things:
1. Straight-up accuracy (did we pick the right winner)
2. The ACTUAL question that matters: does the model beat the closing spread line?
   i.e. when we disagree with the market, are we right often enough to profit after vig?
"""
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss

df = pd.read_parquet("matchup_dataset.parquet")

FEATURES = [
    "home_avg_pts_for_L5", "home_avg_pts_against_L5", "home_avg_margin_L5",
    "home_win_pct_L5", "home_season_win_pct", "home_rest_days",
    "home_qb_changed", "home_qb_starts_with_team",
    "away_avg_pts_for_L5", "away_avg_pts_against_L5", "away_avg_margin_L5",
    "away_win_pct_L5", "away_season_win_pct", "away_rest_days",
    "away_qb_changed", "away_qb_starts_with_team",
    "is_dome", "temp", "wind",
    "spread_line",  # include the market's own line as a feature -- it's the single best predictor there is
]

data = df.dropna(subset=FEATURES + ["home_win"]).copy()
data = data[data["season"] >= 2005].reset_index(drop=True)  # modern era, cleaner data

# Time-based split: train on the past, test on the most recent full season.
# Never randomly shuffle sports data across time -- that leaks future info into training.
train = data[data["season"] <= 2023]
test = data[data["season"] == 2024]

X_train, y_train = train[FEATURES], train["home_win"].astype(int)
X_test, y_test = test[FEATURES], test["home_win"].astype(int)

model = XGBClassifier(
    n_estimators=200, max_depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
)
model.fit(X_train, y_train)

pred_proba = model.predict_proba(X_test)[:, 1]
pred_class = (pred_proba > 0.5).astype(int)

print("=== STRAIGHT-UP WIN PREDICTION (2024 holdout season) ===")
print(f"Accuracy: {accuracy_score(y_test, pred_class):.3f}")
print(f"Log loss: {log_loss(y_test, pred_proba):.3f}")

# --- Convert market spread to an implied win probability for a fair comparison ---
# NOTE: in this table, spread_line is POSITIVE when home is favored (verified empirically
# against home_margin correlation) -- opposite of the usual sportsbook board convention.
# Rough conversion: NFL point spread -> win prob via a logistic curve; ~5.5 pts per
# "unit" of win-prob shift is a standard rough approximation for NFL spreads.
def spread_to_implied_prob(spread):
    return 1 / (1 + np.exp(-spread / 5.5))

test = test.copy()
test["model_prob"] = pred_proba
test["market_implied_prob"] = spread_to_implied_prob(test["spread_line"])
test["edge"] = test["model_prob"] - test["market_implied_prob"]

print("\n=== DOES THE MODEL BEAT THE MARKET? ===")
print(f"Market-implied-prob accuracy (just picking spread favorite): "
      f"{accuracy_score(y_test, (test['market_implied_prob'] > 0.5).astype(int)):.3f}")

# The real test: on games where model disagrees with market by a meaningful margin,
# is the model actually more often right?
threshold = 0.07  # only look at "real" disagreements, not noise
big_edge = test[test["edge"].abs() > threshold].copy()
print(f"\nGames where model diverges from market by >{threshold}: {len(big_edge)} / {len(test)}")

if len(big_edge) > 0:
    # when model says home team more likely than market thinks, did home team win?
    model_favors_home_more = big_edge[big_edge["edge"] > 0]
    model_favors_away_more = big_edge[big_edge["edge"] < 0]

    hit_when_bullish_home = model_favors_home_more["home_win"].mean() if len(model_favors_home_more) else float("nan")
    hit_when_bullish_away = (1 - model_favors_away_more["home_win"]).mean() if len(model_favors_away_more) else float("nan")

    print(f"When model more bullish on HOME than market ({len(model_favors_home_more)} games): "
          f"home actually won {hit_when_bullish_home:.1%} of the time")
    print(f"When model more bullish on AWAY than market ({len(model_favors_away_more)} games): "
          f"away actually won {hit_when_bullish_away:.1%} of the time")
    print(f"\n(Breakeven vs standard -110 vig is 52.4%. Above that = real signal. Below/near = noise.)")

print("\n=== FEATURE IMPORTANCE ===")
importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print(importances.to_string())

model.save_model("model.json")
print("\nModel saved to model.json")