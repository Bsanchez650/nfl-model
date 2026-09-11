"""
Player props model: predict whether a player goes OVER a given stat line
(e.g. receiving yards, rushing yards) based on their own rolling form and matchup.

Unlike team spreads, prop lines are set by books with LESS scrutiny than the
main spread/total -- this is the more realistic place to look for a real edge,
per our earlier discussion. Still needs backtesting before trusting it with money.
"""
import pandas as pd
import numpy as np
import nfl_data_py as nfl
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

STAT_COL = "receiving_yards"   # swap to "rushing_yards", "passing_yards", etc.
POSITION_FILTER = ["WR", "TE"]  # relevant positions for receiving yards


def load_weekly(seasons):
    df = nfl.import_weekly_data(seasons)
    return df


def add_rolling_player_form(df: pd.DataFrame, stat_col: str, window: int = 5) -> pd.DataFrame:
    df = df.sort_values(["player_id", "season", "week"]).copy()
    grp = df.groupby("player_id")

    # shift(1) first so we never use the current game's own result to predict itself
    df[f"{stat_col}_L{window}_avg"] = (
        grp[stat_col].transform(lambda s: s.shift(1).rolling(window, min_periods=2).mean())
    )
    df[f"{stat_col}_L{window}_std"] = (
        grp[stat_col].transform(lambda s: s.shift(1).rolling(window, min_periods=2).std())
    )
    df[f"{stat_col}_season_avg"] = (
        grp[stat_col].transform(lambda s: s.shift(1).expanding().mean())
    )
    # target share / usage trend matters a lot for receiving props
    if "targets" in df.columns:
        df[f"targets_L{window}_avg"] = (
            grp["targets"].transform(lambda s: s.shift(1).rolling(window, min_periods=2).mean())
        )
    return df


def build_prop_dataset(seasons):
    weekly = load_weekly(seasons)
    weekly = weekly[weekly["position"].isin(POSITION_FILTER)].copy()
    weekly = add_rolling_player_form(weekly, STAT_COL, window=5)
    return weekly


if __name__ == "__main__":
    seasons = list(range(2019, 2025))
    df = build_prop_dataset(seasons)

    feature_cols = [
        f"{STAT_COL}_L5_avg", f"{STAT_COL}_L5_std", f"{STAT_COL}_season_avg",
        "targets_L5_avg",
    ]
    df = df.dropna(subset=feature_cols + [STAT_COL])
    print(f"Dataset: {len(df)} player-games with complete rolling features")

    # time split: train through 2023, test on 2024
    train = df[df["season"] <= 2023]
    test = df[df["season"] == 2024]

    X_train, y_train = train[feature_cols], train[STAT_COL]
    X_test, y_test = test[feature_cols], test[STAT_COL]

    model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print(f"\nMAE: {mean_absolute_error(y_test, preds):.2f} yards")
    print(f"Naive baseline MAE (just using L5 avg as the guess): "
          f"{mean_absolute_error(y_test, X_test[f'{STAT_COL}_L5_avg']):.2f} yards")

    # Simulate an "over/under" bet: use the player's own rolling average as a stand-in
    # for what a sportsbook prop line might look like (a real line would come from odds data),
    # then see how often our model's prediction and the actual result agree on over/under.
    test = test.copy()
    test["model_pred"] = preds
    test["pseudo_line"] = X_test[f"{STAT_COL}_L5_avg"]  # proxy for a book's line
    test["model_says_over"] = test["model_pred"] > test["pseudo_line"]
    test["actual_over"] = test[STAT_COL] > test["pseudo_line"]

    agree_rate = (test["model_says_over"] == test["actual_over"]).mean()
    print(f"\nModel direction vs pseudo-line agreement: {agree_rate:.1%}")
    print("(This is directional only -- a real backtest needs actual historical prop lines,")
    print(" which are a separate, harder-to-source data feed than team spreads.)")

    model.save_model("props_model.json")