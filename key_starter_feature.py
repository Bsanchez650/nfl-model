"""
Real feature extraction: for a team, determine which starters (per the depth
chart's ordering) currently have a non-Active injury status.

Key insight from testing: ESPN's depth chart response already embeds each
athlete's current injuries inline, and athletes are ordered by depth within
each position (index 0 = starter). So this needs ONE API call per team --
no separate injuries endpoint or cross-referencing required.
"""
import requests

ESPN_TEAM_IDS = {
    "ARI": 22, "ATL": 1, "BAL": 33, "BUF": 2, "CAR": 29, "CHI": 3, "CIN": 4,
    "CLE": 5, "DAL": 6, "DEN": 7, "DET": 8, "GB": 9, "HOU": 34, "IND": 11,
    "JAX": 30, "KC": 12, "LA": 14, "LAC": 24, "LV": 13, "MIA": 15, "MIN": 16,
    "NE": 17, "NO": 18, "NYG": 19, "NYJ": 20, "PHI": 21, "PIT": 23, "SEA": 26,
    "SF": 25, "TB": 27, "TEN": 10, "WAS": 28,
}

BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team_id}/depthcharts"

# The offensive skill positions that most plausibly move a game's outcome.
# Key names match ESPN's depth chart position keys (seen in test_depth_chart.py output).
# Note some formations have multiple WR slots (wr1/wr2/wr3) -- we check all of them.
KEY_POSITIONS = ["qb", "rb", "wr1", "wr2", "wr3", "te"]

# Any status other than "Active" means some level of concern. We separate
# "Out"/"Doubtful"/"Injured Reserve"/"Suspension" (essentially confirmed absence)
# from "Questionable" (uncertain -- often plays anyway).
CONFIRMED_OUT_STATUSES = {"Out", "Doubtful", "Injured Reserve", "Suspension"}


def get_depth_chart(team_abbr: str):
    team_id = ESPN_TEAM_IDS[team_abbr]
    url = BASE_URL.format(team_id=team_id)
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()


def find_starter(depth_chart_data: dict, position_key: str):
    """
    Depth charts are nested under a list of formations (e.g. 'Base 3-4 D',
    '3WR 1TE'). The same position can appear in more than one formation --
    we just need the first occurrence's starter (index 0 of its athletes list).
    """
    for formation in depth_chart_data.get("depthchart", []):
        positions = formation.get("positions", {})
        if position_key in positions:
            athletes = positions[position_key].get("athletes", [])
            if athletes:
                return athletes[0]  # index 0 = starter
    return None


def key_starter_injury_report(team_abbr: str):
    """
    Returns a list of dicts for each key position's starter who currently
    has a non-Active injury status: name, position, status.
    """
    data = get_depth_chart(team_abbr)
    concerns = []

    for pos_key in KEY_POSITIONS:
        starter = find_starter(data, pos_key)
        if not starter:
            continue

        injuries = starter.get("injuries", [])
        if not injuries:
            continue  # no injury entry at all = presumed healthy

        # Most recent injury entry is what matters (list order not guaranteed,
        # so just take the first -- ESPN typically only lists the current one).
        status = injuries[0].get("status")
        if status and status != "Active":
            concerns.append({
                "position_slot": pos_key,
                "name": starter.get("displayName"),
                "status": status,
                "confirmed_out": status in CONFIRMED_OUT_STATUSES,
            })

    return concerns


def key_starter_out_flag(team_abbr: str) -> int:
    """
    The actual model feature: 1 if ANY key-position starter is confirmed out
    (Out/Doubtful/IR/Suspension), else 0. Questionable alone does not set this,
    since questionable players frequently still play.
    """
    concerns = key_starter_injury_report(team_abbr)
    return int(any(c["confirmed_out"] for c in concerns))


if __name__ == "__main__":
    test_teams = ["SEA", "LV", "KC", "BUF"]

    for team in test_teams:
        print(f"\n=== {team} ===")
        concerns = key_starter_injury_report(team)
        if not concerns:
            print("No key-position starter injury concerns.")
        for c in concerns:
            marker = "CONFIRMED OUT" if c["confirmed_out"] else "questionable"
            print(f"  [{c['position_slot']}] {c['name']} - {c['status']} ({marker})")
        print(f"  --> key_starter_out flag: {key_starter_out_flag(team)}")