# NFL Predictions Pipeline

## Setup (run once)
```
python3 -m venv nflenv
source nflenv/bin/activate      # Windows: nflenv\Scripts\activate
pip install --upgrade pip setuptools wheel
pip install pandas numpy appdirs fastparquet requests
pip install --no-deps nfl_data_py
pip install xgboost scikit-learn fastapi uvicorn
```

## Files
- `build_dataset.py` — pulls historical games/lines from nflverse, builds team-level
  rolling-form features, QB continuity features, and weather features. Run first to
  regenerate `matchup_dataset.parquet`.
- `train_and_backtest.py` — trains the team spread model AND backtests it against
  real closing lines. **Read the printed backtest output before trusting anything
  else in this project.**
- `props_model.py` — player receiving-yards prop model (WR/TE). Directional signal
  only; not backtested against real historical prop lines.
- `weekly_pipeline.py` — pulls the current week's schedule + lines and runs the
  trained model against it. Run this every week during the season.
- `live_weather.py` — live short-range weather forecast for upcoming outdoor games,
  via the free National Weather Service API (no key needed). NOTE: could not be
  tested in the build sandbox (network is locked to GitHub/PyPI only there) --
  will work normally on a machine with regular internet access.
- `api.py` — FastAPI wrapper exposing `/predictions/{season}/{week}` for a Flutter
  app (or anything else) to call over HTTP.

## Running it
```
python3 build_dataset.py          # rebuild historical dataset
python3 train_and_backtest.py     # retrain + see backtest numbers
python3 weekly_pipeline.py        # generate this week's predictions -> weekly_output.json
uvicorn api:app --reload --port 8000   # serve it over HTTP
```

Then: `curl http://localhost:8000/predictions/2026/1`

## Feature history / what's been tried
| Features | Straight-up accuracy | Beats market on disagreements? |
|---|---|---|
| Team rolling form only | 65.8% | No (~27-28% hit rate when disagreeing) |
| + QB continuity (changed/new starter) | 65.8% | No (~19-27%) |
| + Weather (temp/wind/dome) | 66.0% | No (~26-31%) |

Each addition genuinely gets used by the model (shows up in feature importance),
but none of them move the "beats the market" needle. The consistent takeaway:
the closing spread line already prices in QB changes and weather about as well
as this model does with the same public data. This is expected, not a bug --
see the project's original scoping conversation for why.

## The honest bottom line
- Team spread model: solid straight-up accuracy (~66%), but worse than just
  following the market (~70%), and wrong more often than right when it disagrees
  with the closing line. Does not have a real edge over the market.
- Player props model: weak positive signal (56.6% directional agreement vs. a
  proxy line), unverified against real book lines.
- This is a legitimate decision-support / tracking tool and a solid ML project.
  It is not a proven source of betting profit, and nothing here is financial advice.

## Dashboard
`dashboard.html` is a standalone page — no build step, just open it in a browser
while your API is running:

1. `uvicorn api:app --reload --port 8000` (in one terminal, leave it running)
2. Double-click `dashboard.html` to open it in your browser (or drag it into a tab)
3. Set season/week, hit Load

It talks to `http://localhost:8000` directly from the page's JavaScript — nothing
to deploy, just keep the API running in the background while you use it.

## Tracking your results
Every time you hit `/predictions/{season}/{week}` (or run `weekly_pipeline.py`
directly), that week's picks get appended to `predictions_log.json`. On each run,
any previously-logged game that now has a final score gets automatically graded
(correct/incorrect) — no manual scoring needed.

- `GET /track_record` → your running accuracy across every graded game this season
- The dashboard shows this at the top automatically
- `predictions_log.json` is plain JSON if you want to look at it directly or load
  it into a spreadsheet later

## Score prediction
`score_model.py` trains two regressors (home score, away score) on the same
features as everything else. Run it once after `train_and_backtest.py`:
```
python3 score_model.py
```
This saves `home_score_model.json` / `away_score_model.json`, which
`weekly_pipeline.py` automatically picks up and includes in its output.

**Honest backtest result:** the model's predicted scores are about tied with what
you'd get by algebraically backing out the score from the market's own closing
spread + total (within ~0.05 points MAE) — same pattern as every other feature
we've tested. It's a real, non-fabricated number, just not an edge over the market.

## Dashboard v2
Redesigned to a light, ESPN-scoreboard-style layout: team-colored badges, predicted
final score per team, and a two-color win-probability bar split by team color (the
same visual ESPN's own gamecast win-probability bar uses). Nothing to install —
same as before, just open `dashboard.html` while the API is running.

## Finished vs. upcoming games on the dashboard
Games that have already been played now show their real final score and a
"MODEL RIGHT / MODEL WRONG" tag (with the model's original prediction shown
below for comparison), instead of always displaying a prediction. Games that
haven't happened yet still show the predicted score and win-probability bar.

Note: `weekly_pipeline.py` now prints a clear warning if `home_score_model.json`
/ `away_score_model.json` are missing, instead of silently skipping score
predictions -- if you ever see "--" for scores, check the terminal output from
your last `python weekly_pipeline.py` run for that message.

## Tie-breaking on predicted scores
The two score models predict home/away points independently, so occasionally
both round to the same integer (e.g. 20.96 and 20.55 both round to 21) --
displaying an impossible tie. Fixed: when rounding produces a tie,
`weekly_pipeline.py` now nudges the score that was actually higher *before*
rounding up by 1 point, so the predicted winner always matches the underlying
prediction.

## Keeping it running
- **Server**: create `run_server.bat` with `activate` + `uvicorn` commands, double-click
  to start, minimize instead of closing. Optionally drop a shortcut in your Windows
  Startup folder (`shell:startup`) to auto-start on login.
- **Weekly updates**: create `run_weekly.bat` that activates the venv and runs
  `python weekly_pipeline.py`, then schedule it daily via Windows Task Scheduler so
  predictions/grading refresh automatically without you running it by hand.
- **Dashboard**: now auto-refreshes every 10 minutes on its own while the tab is open.

## Deploying to the cloud (access from anywhere, PC can be off)
Files added: `requirements.txt`, `render.yaml`.

1. Train locally one last time (`build_dataset.py` → `train_and_backtest.py` → `score_model.py`)
   so the model files are current -- the cloud server uses these as-is, it doesn't retrain itself.
2. Push everything to a GitHub repo.
3. On render.com: New → Blueprint → connect the repo. It reads `render.yaml` and
   deploys the API and the dashboard as two free services automatically.
4. Copy your API's Render URL, paste it into `API_BASE` in `dashboard.html`, push again.
5. Open the dashboard's Render URL on your phone, anywhere, no WiFi/PC needed.

**Caveats of the free tier:**
- Sleeps after 15 min idle; first request after that takes 30-60 sec to wake up.
- No guaranteed persistent disk -- `predictions_log.json` (your season accuracy
  tracker) may reset on restarts/redeploys. Fine for casual use; ask if you want
  this made durable (small free database instead) since it's extra setup.
