"""
Build a training dataset for NFL game outcome prediction.

Source: nflverse games.csv (schedules, scores, closing lines, weather, rest days)
Output: one row per game, with pre-game features only (no leakage from final score),
        plus the actual result and the closing spread line for backtesting.
"""
import pandas as pd
import numpy as np

GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"


def load_games():
    df = pd.read_csv(GAMES_URL)
    return df


def build_team_game_log(games: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape from one-row-per-game (home/away columns) to one-row-per-team-per-game,
    so we can compute rolling form for each team regardless of home/away.
    """
    home = games.rename(columns={
        "home_team": "team", "away_team": "opponent",
        "home_score": "points_for", "away_score": "points_against",
        "home_rest": "rest_days", "home_moneyline": "moneyline",
        "home_qb_id": "qb_id", "home_coach": "coach",
    }).copy()
    home["is_home"] = 1

    away = games.rename(columns={
        "away_team": "team", "home_team": "opponent",
        "away_score": "points_for", "home_score": "points_against",
        "away_rest": "rest_days", "away_moneyline": "moneyline",
        "away_qb_id": "qb_id", "away_coach": "coach",
    }).copy()
    away["is_home"] = 0

    keep_cols = ["game_id", "season", "week", "gameday", "team", "opponent",
                 "points_for", "points_against", "is_home", "rest_days",
                 "moneyline", "qb_id", "coach", "div_game", "roof", "surface", "temp", "wind"]
    long_df = pd.concat([home[keep_cols], away[keep_cols]], ignore_index=True)
    long_df["gameday"] = pd.to_datetime(long_df["gameday"])
    long_df = long_df.sort_values(["team", "gameday"]).reset_index(drop=True)

    long_df["win"] = (long_df["points_for"] > long_df["points_against"]).astype(float)
    long_df.loc[long_df["points_for"].isna(), "win"] = np.nan
    long_df["margin"] = long_df["points_for"] - long_df["points_against"]

    return long_df


def add_weather_features(games: pd.DataFrame) -> pd.DataFrame:
    """
    Weather features, straight off the actual game record (not a live forecast --
    that's a separate live-only step, see get_live_weather_forecast below).

    - is_dome: 1 if the game was played indoors, in which case temp/wind are
      structurally irrelevant (always ~70F, no wind) -- this flag alone often
      matters more to the model than the raw numbers.
    - temp / wind: only meaningful for outdoor games, left as NaN otherwise so the
      model can learn "dome = neutral" rather than being fed a fake average temp.
    """
    games = games.copy()
    games["is_dome"] = games["roof"].isin(["dome", "closed"]).astype(float)
    # Dome/closed-roof games have no recorded temp/wind (there's no weather indoors) --
    # fill with a neutral constant so is_dome carries that signal instead of dropna()
    # silently discarding every dome game from training. Outdoor games with genuinely
    # missing data (older seasons, pre-tracking) stay NaN and get dropped as usual.
    games.loc[games["is_dome"] == 1, "temp"] = games.loc[games["is_dome"] == 1, "temp"].fillna(70)
    games.loc[games["is_dome"] == 1, "wind"] = games.loc[games["is_dome"] == 1, "wind"].fillna(0)
    return games


def add_rolling_form(long_df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    For each team-game, compute rolling averages using ONLY prior games
    (shift(1) before rolling) so there's no data leakage from the game being predicted.
    """
    long_df = long_df.sort_values(["team", "gameday"])
    grp = long_df.groupby("team")

    for col, new_name in [
        ("points_for", "avg_pts_for"),
        ("points_against", "avg_pts_against"),
        ("margin", "avg_margin"),
        ("win", "win_pct"),
    ]:
        long_df[f"{new_name}_L{window}"] = (
            grp[col].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )

    # season-to-date form (all prior games this season)
    long_df["season_win_pct"] = (
        long_df.groupby(["team", "season"])["win"]
        .transform(lambda s: s.shift(1).expanding().mean())
    )

    return long_df


def add_qb_continuity_features(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add two QB-aware features that the plain team rolling average misses:
      - qb_changed: did the starting QB change from this team's previous game?
      - qb_starts_with_team: how many career starts does this specific QB have
        with THIS team (low number = new arrival, small sample, less reliable
        team-level history applies to him).
    This doesn't need separate passing-stats data -- qb_id is already in games.csv,
    so continuity can be computed directly from the same team-game log.
    """
    long_df = long_df.sort_values(["team", "gameday"]).copy()

    prev_qb = long_df.groupby("team")["qb_id"].shift(1)
    long_df["qb_changed"] = (long_df["qb_id"] != prev_qb).astype(float)
    long_df.loc[prev_qb.isna(), "qb_changed"] = np.nan  # first known game for a team: unknown, not "changed"

    # count this QB's prior starts specifically with this team (not career-wide) --
    # shift(1) so the game being predicted doesn't count itself
    long_df["qb_starts_with_team"] = (
        long_df.groupby(["team", "qb_id"]).cumcount()
    )

    return long_df



def assemble_matchup_dataset(long_df: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """
    Rejoin the team-game-log (with rolling features) back into one row per game:
    home team's features vs away team's features, plus the actual closing line/result.
    """
    feature_cols = [
        "avg_pts_for_L5", "avg_pts_against_L5", "avg_margin_L5", "win_pct_L5",
        "season_win_pct", "rest_days", "qb_changed", "qb_starts_with_team",
    ]

    home_feats = long_df[long_df["is_home"] == 1][["game_id", "team"] + feature_cols]
    home_feats = home_feats.rename(columns={c: f"home_{c}" for c in feature_cols})

    away_feats = long_df[long_df["is_home"] == 0][["game_id", "team"] + feature_cols]
    away_feats = away_feats.rename(columns={c: f"away_{c}" for c in feature_cols})

    merged = games.merge(home_feats, on="game_id", how="left", suffixes=("", "_h"))
    merged = merged.merge(away_feats, on="game_id", how="left", suffixes=("", "_a"))

    # target: did the home team cover the closing spread? and who won straight-up
    merged["home_win"] = (merged["home_score"] > merged["away_score"]).astype("Int64")
    merged["home_margin"] = merged["home_score"] - merged["away_score"]
    # NOTE: in this nflverse table, spread_line is POSITIVE when the home team is favored
    # (verified empirically: positive correlation with home_margin) -- opposite of the
    # typical sportsbook board convention where favorites are shown negative.
    # Home "covers" if their actual margin exceeds the points they were favored by.
    merged["home_covered"] = (merged["home_margin"] - merged["spread_line"] > 0).astype("Int64")

    return merged


if __name__ == "__main__":
    games = load_games()
    print(f"Loaded {len(games)} games, seasons {games['season'].min()}-{games['season'].max()}")

    games = add_weather_features(games)
    long_df = build_team_game_log(games)
    long_df = add_rolling_form(long_df, window=5)
    long_df = add_qb_continuity_features(long_df)
    dataset = assemble_matchup_dataset(long_df, games)

    dataset.to_parquet("matchup_dataset.parquet")
    print(f"Saved matchup dataset: {dataset.shape}")
    print("\nSample columns:", [c for c in dataset.columns if "avg_" in c or "win_pct" in c or c in
                                 ["home_covered", "home_win", "spread_line"]])

    complete = dataset.dropna(subset=["home_avg_margin_L5", "away_avg_margin_L5", "home_score"])
    print(f"\nRows with complete features + known result: {len(complete)}")
    print(complete[["season", "week", "home_team", "away_team", "spread_line",
                     "home_margin", "home_covered"]].tail(10))