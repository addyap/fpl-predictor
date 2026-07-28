"""
data_fetch.py
=============
All external data access for the FPL predictor lives here.

Three sources are used:
1. football-data.co.uk  -> free historical CSVs (used to TRAIN the model)
2. football-data.org    -> current-season fixtures/results (needs an API key)
3. Official FPL API     -> players, prices, ownership, teams (no key needed)

Every network helper is defensive: on failure it returns an empty/neutral
value and lets the caller decide how to degrade gracefully, rather than
raising and crashing the Streamlit app.
"""

from __future__ import annotations

import io
import os
from typing import Callable, Optional

import pandas as pd
import requests

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# football-data.co.uk premier-league CSVs. E0 == English Premier League.
# Season codes are the two-year form, e.g. 2526 == 2025/26.
FOOTBALL_DATA_CO_UK = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
HISTORICAL_SEASONS = ["2425", "2526"]  # two most recent completed seasons
# Tag derived from the season list so changing seasons invalidates cached CSVs
# (old files have a different name and are simply re-downloaded).
SEASON_TAG = "".join(HISTORICAL_SEASONS)

# football-data.org v4 REST API
FOOTBALL_DATA_ORG = "https://api.football-data.org/v4"
PL_COMPETITION = "PL"  # Premier League competition code

# Official (unofficial but public) Fantasy Premier League endpoints
FPL_BASE = "https://fantasy.premierleague.com/api"

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


# --------------------------------------------------------------------------- #
# League catalogue
# --------------------------------------------------------------------------- #
# ``code`` is football-data.co.uk's division code (drives the historical CSV
# used for TRAINING). ``org`` is football-data.org's competition code (drives
# LIVE upcoming fixtures) — None when the free tier doesn't serve that league,
# in which case the UI falls back to a model-based team power ranking.
LEAGUES = [
    {"code": "E0",  "name": "Premier League (England)",      "org": "PL"},
    {"code": "E1",  "name": "Championship (England)",         "org": "ELC"},
    {"code": "E2",  "name": "League One (England)",           "org": None},
    {"code": "E3",  "name": "League Two (England)",           "org": None},
    {"code": "EC",  "name": "National League (England)",      "org": None},
    {"code": "SC0", "name": "Premiership (Scotland)",         "org": None},
    {"code": "SC1", "name": "Championship (Scotland)",        "org": None},
    {"code": "SC2", "name": "League One (Scotland)",          "org": None},
    {"code": "SC3", "name": "League Two (Scotland)",          "org": None},
    {"code": "D1",  "name": "Bundesliga (Germany)",           "org": "BL1"},
    {"code": "D2",  "name": "2. Bundesliga (Germany)",        "org": None},
    {"code": "SP1", "name": "La Liga (Spain)",                "org": "PD"},
    {"code": "SP2", "name": "La Liga 2 (Spain)",              "org": None},
    {"code": "I1",  "name": "Serie A (Italy)",                "org": "SA"},
    {"code": "I2",  "name": "Serie B (Italy)",                "org": None},
    {"code": "F1",  "name": "Ligue 1 (France)",               "org": "FL1"},
    {"code": "F2",  "name": "Ligue 2 (France)",               "org": None},
    {"code": "N1",  "name": "Eredivisie (Netherlands)",       "org": "DED"},
    {"code": "B1",  "name": "First Division A (Belgium)",     "org": None},
    {"code": "P1",  "name": "Primeira Liga (Portugal)",       "org": "PPL"},
    {"code": "T1",  "name": "Süper Lig (Turkey)",             "org": None},
    {"code": "G1",  "name": "Super League (Greece)",          "org": None},
]

# Convenience lookups
LEAGUE_BY_CODE = {lg["code"]: lg for lg in LEAGUES}
PL_CODE = "E0"


def get_league(code: str) -> dict:
    """Return the league catalogue entry for a football-data.co.uk code."""
    return LEAGUE_BY_CODE.get(code, {"code": code, "name": code, "org": None})


def historical_path(code: str) -> str:
    """
    Filesystem path of the cached historical CSV for a league code. The season
    tag is baked into the filename so bumping HISTORICAL_SEASONS transparently
    invalidates old caches (they no longer match and get re-downloaded).
    """
    return os.path.join(DATA_DIR, f"historical_{code}_{SEASON_TAG}.csv")

# The subset of football-data.co.uk columns we actually need. Keeping this
# explicit means a schema change upstream fails loudly rather than silently
# feeding junk into the model.
CORE_COLUMNS = {
    "Date": "date",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "home_goals",   # full-time home goals
    "FTAG": "away_goals",   # full-time away goals
    "FTR": "result",        # H / D / A
}

# Bookmaker odds columns (Bet365) we keep when present for the odds tab.
ODDS_COLUMNS = {"B365H": "odds_home", "B365D": "odds_draw", "B365A": "odds_away"}

REQUEST_TIMEOUT = 20  # seconds


# --------------------------------------------------------------------------- #
# Historical data (training set)
# --------------------------------------------------------------------------- #

def historical_exists(code: str = PL_CODE) -> bool:
    """True if the historical CSV for ``code`` has already been downloaded."""
    return os.path.exists(historical_path(code))


def all_historical_exists() -> bool:
    """True only when every catalogued league has a cached CSV."""
    return all(historical_exists(lg["code"]) for lg in LEAGUES)


def download_historical(
    code: str = PL_CODE,
    progress: Optional[Callable[[float, str], None]] = None,
) -> pd.DataFrame:
    """
    Download the configured seasons for one league from football-data.co.uk,
    keep the columns we care about, merge them and cache to
    ``data/historical_<code>.csv``.

    ``progress`` is an optional callback ``(fraction, message)`` used to drive
    a Streamlit progress bar. It is called between 0.0 and 1.0.

    Returns the merged DataFrame. Raises RuntimeError if nothing could be
    downloaded (so the caller can surface a clear message).
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    league = get_league(code)
    frames: list[pd.DataFrame] = []
    total = len(HISTORICAL_SEASONS)

    for i, season in enumerate(HISTORICAL_SEASONS):
        pretty = f"20{season[:2]}/{season[2:]}"
        if progress:
            progress(i / total, f"Downloading {league['name']} {pretty}…")

        url = FOOTBALL_DATA_CO_UK.format(season=season, code=code)
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
        except requests.RequestException as exc:  # network / HTTP error
            # Skip this season but keep going; we may still have another.
            print(f"[data_fetch] failed to download {url}: {exc}")
            continue

        # football-data.co.uk mixes encodings across files (some UTF-8, some
        # Windows-1252). Decode from raw bytes trying the likeliest first so
        # accented names (e.g. "Preußen Münster") come through intact.
        raw = None
        for enc in ("utf-8-sig", "cp1252", "latin-1"):
            try:
                raw = pd.read_csv(io.BytesIO(resp.content), encoding=enc)
                break
            except Exception:  # wrong encoding or parse error — try the next
                raw = None
                continue
        if raw is None:
            print(f"[data_fetch] failed to parse {url}")
            continue

        keep = {**CORE_COLUMNS, **{k: v for k, v in ODDS_COLUMNS.items() if k in raw.columns}}
        present = [c for c in keep if c in raw.columns]
        if not all(c in raw.columns for c in CORE_COLUMNS):
            print(f"[data_fetch] {league['name']} {pretty} missing core columns, skipping")
            continue

        df = raw[present].rename(columns=keep)
        df["season"] = pretty
        frames.append(df)

    if progress:
        progress(1.0, "Finalising…")

    if not frames:
        raise RuntimeError(
            f"Could not download historical data for {league['name']} "
            f"from football-data.co.uk."
        )

    merged = pd.concat(frames, ignore_index=True)
    # Normalise types
    merged["date"] = pd.to_datetime(merged["date"], dayfirst=True, errors="coerce")
    merged = merged.dropna(subset=["date", "home_team", "away_team", "result"])
    merged = merged.sort_values("date").reset_index(drop=True)
    merged.to_csv(historical_path(code), index=False)
    return merged


def download_all_historical(
    progress: Optional[Callable[[float, str], None]] = None,
) -> dict[str, pd.DataFrame]:
    """
    Download every catalogued league. Returns ``{code: DataFrame}`` for the
    leagues that succeeded (a league with no data upstream is skipped, not
    fatal). ``progress`` spans all leagues 0.0 -> 1.0.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    results: dict[str, pd.DataFrame] = {}
    n = len(LEAGUES)
    for i, lg in enumerate(LEAGUES):
        base = i / n
        span = 1.0 / n

        def _sub(frac: float, msg: str, _b=base, _s=span):
            if progress:
                progress(_b + frac * _s, msg)

        try:
            results[lg["code"]] = download_historical(lg["code"], progress=_sub)
        except RuntimeError as exc:
            print(f"[data_fetch] skipping {lg['name']}: {exc}")
            continue
    if progress:
        progress(1.0, "All leagues downloaded.")
    return results


def load_historical(code: str = PL_CODE) -> pd.DataFrame:
    """
    Load the cached historical CSV for a league. Downloads it first if missing.
    (No progress bar here — used by the cached model loader.)
    """
    if not historical_exists(code):
        return download_historical(code)
    df = pd.read_csv(historical_path(code))
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


# --------------------------------------------------------------------------- #
# football-data.org (current season)
# --------------------------------------------------------------------------- #

def _org_headers(api_key: str) -> dict:
    return {"X-Auth-Token": api_key}


def validate_org_key(api_key: str) -> tuple[bool, str]:
    """
    Cheaply verify a football-data.org key. Returns ``(ok, message)``.
    A 200 means the key works; 400/403 means it is invalid/forbidden.
    """
    if not api_key or not api_key.strip():
        return False, "No API key provided."
    try:
        resp = requests.get(
            f"{FOOTBALL_DATA_ORG}/competitions/{PL_COMPETITION}",
            headers=_org_headers(api_key.strip()),
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        return False, f"Network error: {exc}"

    if resp.status_code == 200:
        return True, "API key valid."
    if resp.status_code in (400, 403):
        return False, "API key rejected (invalid or unauthorised)."
    if resp.status_code == 429:
        return False, "Rate limit reached. Wait a minute and retry."
    return False, f"Unexpected response ({resp.status_code})."


def fetch_upcoming_matches(
    api_key: str, competition: str = PL_COMPETITION, limit: int = 10
) -> pd.DataFrame:
    """
    Return the next scheduled matches for a competition as a DataFrame with
    columns ``home_team``, ``away_team``, ``utc_date``. Empty on failure or
    when ``competition`` is None (league not on the free tier).
    """
    cols = ["home_team", "away_team", "utc_date"]
    if not competition:
        return pd.DataFrame(columns=cols)
    try:
        resp = requests.get(
            f"{FOOTBALL_DATA_ORG}/competitions/{competition}/matches",
            headers=_org_headers(api_key.strip()),
            params={"status": "SCHEDULED"},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        matches = resp.json().get("matches", [])
    except (requests.RequestException, ValueError) as exc:
        print(f"[data_fetch] fetch_upcoming_matches failed: {exc}")
        return pd.DataFrame(columns=cols)

    rows = []
    for m in matches[:limit]:
        rows.append(
            {
                "home_team": m.get("homeTeam", {}).get("name", "N/A"),
                "away_team": m.get("awayTeam", {}).get("name", "N/A"),
                "utc_date": m.get("utcDate", "N/A"),
            }
        )
    return pd.DataFrame(rows, columns=cols)


def fetch_recent_results(
    api_key: str, competition: str = PL_COMPETITION, limit: int = 40
) -> pd.DataFrame:
    """
    Most recent FINISHED results for a competition, used to refresh recent
    form. Columns match the historical schema so features.py treats them
    uniformly. Empty on failure or when ``competition`` is None.
    """
    cols = ["date", "home_team", "away_team", "home_goals", "away_goals", "result"]
    if not competition:
        return pd.DataFrame(columns=cols)
    try:
        resp = requests.get(
            f"{FOOTBALL_DATA_ORG}/competitions/{competition}/matches",
            headers=_org_headers(api_key.strip()),
            params={"status": "FINISHED"},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        matches = resp.json().get("matches", [])
    except (requests.RequestException, ValueError) as exc:
        print(f"[data_fetch] fetch_recent_results failed: {exc}")
        return pd.DataFrame(columns=cols)

    rows = []
    for m in matches[-limit:]:
        score = m.get("score", {}).get("fullTime", {})
        hg, ag = score.get("home"), score.get("away")
        if hg is None or ag is None:
            continue
        winner = m.get("score", {}).get("winner")
        result = {"HOME_TEAM": "H", "AWAY_TEAM": "A", "DRAW": "D"}.get(winner, "D")
        rows.append(
            {
                "date": m.get("utcDate"),
                "home_team": m.get("homeTeam", {}).get("name", "N/A"),
                "away_team": m.get("awayTeam", {}).get("name", "N/A"),
                "home_goals": hg,
                "away_goals": ag,
                "result": result,
            }
        )
    df = pd.DataFrame(rows, columns=cols)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


# --------------------------------------------------------------------------- #
# Fantasy Premier League API (no key required)
# --------------------------------------------------------------------------- #

def fetch_fpl_bootstrap() -> dict:
    """
    The FPL 'bootstrap-static' payload: all players, teams, positions and the
    current gameweek. Returns {} on failure.
    """
    try:
        resp = requests.get(f"{FPL_BASE}/bootstrap-static/", timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except (requests.RequestException, ValueError) as exc:
        print(f"[data_fetch] fetch_fpl_bootstrap failed: {exc}")
        return {}


def fetch_fpl_fixtures() -> list[dict]:
    """All FPL fixtures (used for fixture-difficulty). Returns [] on failure."""
    try:
        resp = requests.get(f"{FPL_BASE}/fixtures/", timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except (requests.RequestException, ValueError) as exc:
        print(f"[data_fetch] fetch_fpl_fixtures failed: {exc}")
        return []


def fetch_fpl_entry(team_id: int) -> dict:
    """
    Fetch a manager's team metadata (name, manager, current event). Works even
    before the season starts, unlike the picks endpoint. Returns {} on failure
    or an invalid id, so callers can distinguish "no such team" from "team
    exists but squad picks aren't published yet (pre-season)".
    """
    if not team_id:
        return {}
    try:
        resp = requests.get(
            f"{FPL_BASE}/entry/{int(team_id)}/", timeout=REQUEST_TIMEOUT
        )
        resp.raise_for_status()
        return resp.json()
    except (requests.RequestException, ValueError, TypeError) as exc:
        print(f"[data_fetch] fetch_fpl_entry({team_id}) failed: {exc}")
        return {}


def fetch_fpl_team(team_id: int) -> dict:
    """
    Fetch a manager's current squad. Uses the latest finished/current event
    'picks' endpoint. Returns {} on failure or invalid id.
    """
    if not team_id:
        return {}
    try:
        # Find current event from bootstrap, then fetch that event's picks.
        boot = fetch_fpl_bootstrap()
        events = boot.get("events", [])
        current = next((e["id"] for e in events if e.get("is_current")), None)
        if current is None:
            current = next((e["id"] for e in events if e.get("is_next")), 1)

        resp = requests.get(
            f"{FPL_BASE}/entry/{int(team_id)}/event/{current}/picks/",
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except (requests.RequestException, ValueError, TypeError) as exc:
        print(f"[data_fetch] fetch_fpl_team({team_id}) failed: {exc}")
        return {}
