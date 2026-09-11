"""
Live weather forecast for upcoming outdoor games.

Uses the National Weather Service API (api.weather.gov) -- free, no API key
required, official US government forecast data. Only meaningful for outdoor
games; dome/closed-roof games are always climate-controlled.

NOTE: this sandbox's network is locked to a small allowlist (GitHub, PyPI, etc.)
for security and cannot reach api.weather.gov to test this live. Run this on
your own machine, where normal internet access works, to actually use it.
"""
import requests

# Approximate lat/lon for each team's home stadium (outdoor stadiums only need
# to be accurate here -- dome teams never need a forecast).
STADIUM_COORDS = {
    "BUF": (42.7738, -78.7870), "GB": (44.5013, -88.0622), "CHI": (41.8623, -87.6167),
    "CLE": (41.5061, -81.6995), "CIN": (39.0955, -84.5160), "PIT": (40.4468, -80.0158),
    "BAL": (39.2780, -76.6227), "NE": (42.0909, -71.2643), "NYJ": (40.8135, -74.0745),
    "NYG": (40.8135, -74.0745), "PHI": (39.9008, -75.1675), "WAS": (38.9078, -76.8645),
    "CAR": (35.2258, -80.8528), "TB": (27.9759, -82.5033), "JAX": (30.3239, -81.6373),
    "TEN": (36.1665, -86.7713), "DEN": (39.7439, -105.0201), "KC": (39.0489, -94.4839),
    "SEA": (47.5952, -122.3316), "SF": (37.4030, -121.9700),
    # dome/closed-roof teams intentionally omitted -- forecast is irrelevant indoors
}


def get_live_weather_forecast(team_abbr: str, game_datetime_iso: str) -> dict | None:
    """
    Returns a short-range forecast for an outdoor stadium, or None for dome teams
    or teams not in the coords table. NWS forecasts are only reliable ~7 days out,
    which is fine since you'd run this the week of the game, not months ahead.
    """
    coords = STADIUM_COORDS.get(team_abbr)
    if coords is None:
        return None  # dome team, or missing from the table -- no forecast needed/available

    lat, lon = coords
    points_resp = requests.get(f"https://api.weather.gov/points/{lat},{lon}", timeout=10)
    points_resp.raise_for_status()
    forecast_url = points_resp.json()["properties"]["forecastHourly"]

    forecast_resp = requests.get(forecast_url, timeout=10)
    forecast_resp.raise_for_status()
    periods = forecast_resp.json()["properties"]["periods"]

    # find the forecast period closest to kickoff time
    from datetime import datetime
    kickoff = datetime.fromisoformat(game_datetime_iso)
    closest = min(periods, key=lambda p: abs(
        datetime.fromisoformat(p["startTime"]) - kickoff))

    return {
        "temperature_f": closest["temperature"],
        "wind_speed": closest["windSpeed"],
        "wind_direction": closest["windDirection"],
        "short_forecast": closest["shortForecast"],
    }


if __name__ == "__main__":
    # Example: Buffalo's outdoor stadium in late season is the classic "check the weather" case
    forecast = get_live_weather_forecast("BUF", "2026-12-20T13:00:00")
    print(forecast)
