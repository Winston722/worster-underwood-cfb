# worster-underwood-cfb

An ensemble college football ranking system combining two independent models: **Underwood** (power ratings via regression) and **Worster** (résumé rankings via win/loss record). Results are published daily to Google Sheets during the season.

---

## Project Structure

```
worster_underwood_cfb/
    data/
        cfbd_api.py       # CFBD API fetcher with on-disk cache
    transform/
        schedule.py       # HFA adjustment, weighting
    models/
        underwood.py      # Power rating model
        worster.py        # Résumé ranking model
batch/
    main.py               # CLI entry point
    model.py              # Shapes model outputs into DataFrames
    sheets.py             # Google Sheets writer
    database.py           # Database writer (stub, not yet implemented)
```

---

## Models

### Underwood (Power Ratings)

A weighted least-squares regression model that estimates team strength from margin of victory, adjusted for home field advantage and recency.

**Pipeline:**

1. `prepare_schedule()` — normalizes raw game data into `[week, winner, loser, hfa_margin]`. Subtracts a home field advantage constant (default 3 pts) from the home team's margin so margins reflect true team quality. Postseason games are mapped to week 18.
2. `add_weight()` — assigns each game a weight based on recency and how many FBS games each participating team has played. The game-count factor implicitly down-weights FCS opponents. Weights are normalized to sum to 100.
3. `get_initial()` — runs a single global sparse WLS fit across all games simultaneously. Each game is a row in a sparse design matrix (winner column = +1, loser = -1); the target is the HFA-adjusted margin. Solved via `scipy.sparse.linalg.lsqr` (minimum-norm solution).
4. `get_rating()` — refines each team's rating individually using the initial global ratings as fixed opponent baselines. Equivalent to a weighted mean of `hfa_margin + opponent_rating` across all of a team's games.
5. `get_ratings()` — runs steps 3–4 for all teams and returns a DataFrame sorted descending by rating.
6. `get_error()` — computes per-team prediction RMSE, then applies Bayesian shrinkage toward a prior of ~6pt RMSE (pseudo-count of 22 games, approximately a full FBS regular season). Teams with fewer games are pulled toward the prior; full-schedule teams are barely affected.
7. `combined()` — joins ratings and error into the final output: `[team, rating, pseudo_sd]`.

**Output columns:** `Rank, Team, Rating, Std Dev`

**Not yet implemented:** `error_hfa()` and `error_decay()` — parameter optimization stubs intended for use with `scipy.optimize` to find the optimal HFA constant and time-decay exponent.

---

### Worster (Résumé Rankings)

A pure win/loss résumé system with no margin of victory. Teams are sorted by a cascading key that rewards both the quality of teams you've beaten and penalizes you for losses to weak teams.

**Sort priority (cascading):**
1. Wins (descending)
2. For each depth `i` from 1 to K (default 15):
   - `wins_from_ibest` — win total of the i-th best opponent you defeated (descending)
   - `wins_from_iworst` — win total of the i-th worst team that defeated you (descending)
3. Last year's equivalent metrics as a final tiebreaker

Ties are dropped (a tie produces neither a win nor a loss and cannot be represented cleanly in a pure win/loss system). FCS-only schedules are included in the raw data but FBS-only output is filtered in `batch/model.py`.

**Output columns:** `Rank, team, wins, losses, wins_from_1best, wins_from_1worst, ..., ly_wins, ly_losses, ...`

---

## Data

Game data is fetched from the [College Football Data API](https://api.collegefootballdata.com) (CFBD). Both the current season and prior season are fetched (prior year is used as a Worster tiebreaker).

**Caching:** Results are cached to disk as Parquet (falling back to Pickle if pyarrow is unavailable) with a 24-hour TTL. Cache location defaults to `.cache/cfbd/` or the `WU_CFB_CACHE_DIR` environment variable.

---

## Batch Pipeline

`batch/main.py` is the CLI entry point:

```
python -m batch.main                 # current year, use cache
python -m batch.main --year 2025     # specific year
python -m batch.main --force-refresh # bypass cache, re-fetch from API
```

**Flow:** fetch data → run Underwood + Worster → filter to FBS → write to Google Sheets

Google Sheets tabs:
- `Underwood Shuttle` — power ratings
- `Worster Shuttle` — résumé rankings
- `Upcoming Shuttle` — unplayed games involving at least one FBS team

---

## Setup

**Required environment variables** (in `.env`):

```
CFBD_API_KEY=<your College Football Data API key>
GOOGLE_SERVICE_ACCOUNT_FILE=<path to service account JSON>
# OR
GOOGLE_SERVICE_ACCOUNT_JSON=<full service account JSON as a single-line string>
```

**Install dependencies:**

```
uv sync
uv sync --extra sheets    # add Google Sheets support
uv sync --extra parquet   # add Parquet cache support
```
