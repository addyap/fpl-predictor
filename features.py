"""
features.py
===========
Turns raw match rows into the numeric features the model trains and predicts on.

Feature set (per fixture, home-team-centric):
    recent_form         home recent form minus away recent form (points/game, last N)
    h2h                 historical head-to-head result bias (home perspective)
    home_goals_for      home team's avg goals scored (at home, rolling)
    home_goals_against  home team's avg goals conceded (at home, rolling)
    away_goals_for      away team's avg goals scored (away, rolling)
    away_goals_against  away team's avg goals conceded (away, rolling)

The same routine builds the training matrix from the historical CSV and a
single-row feature vector for an upcoming fixture, so training and inference
never drift apart.
"""

from __future__ import annotations

import pandas as pd

FEATURE_COLUMNS = [
    "recent_form",
    "h2h",
    "home_goals_for",
    "home_goals_against",
    "away_goals_for",
    "away_goals_against",
]

FORM_WINDOW = 6      # matches used for recent form
ROLLING_WINDOW = 10  # matches used for rolling goal averages


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _points(result: str, is_home: bool) -> int:
    """League points a team earned from a result, from its own perspective."""
    if result == "D":
        return 1
    if (result == "H" and is_home) or (result == "A" and not is_home):
        return 3
    return 0


def _team_match_history(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Build, for each team, a chronological frame of its matches with the points
    it earned and goals for/against. Used to compute pre-match rolling stats.
    """
    records: dict[str, list[dict]] = {}
    for _, row in df.iterrows():
        h, a = row["home_team"], row["away_team"]
        hg, ag = row["home_goals"], row["away_goals"]
        res = row["result"]
        records.setdefault(h, []).append(
            {"date": row["date"], "pts": _points(res, True), "gf": hg, "ga": ag, "home": True}
        )
        records.setdefault(a, []).append(
            {"date": row["date"], "pts": _points(res, False), "gf": ag, "ga": hg, "home": False}
        )
    return {t: pd.DataFrame(r).sort_values("date").reset_index(drop=True) for t, r in records.items()}


def _pre_match_stats(hist: pd.DataFrame, before_idx: int, home_only: bool | None) -> dict:
    """
    Average form / goals for a team using only matches strictly BEFORE
    ``before_idx`` (prevents leakage). ``home_only`` filters to home or away
    games; None uses all games (for form).
    """
    past = hist.iloc[:before_idx]
    if home_only is not None:
        past = past[past["home"] == home_only]
    if past.empty:
        return {"form": 1.0, "gf": 1.2, "ga": 1.2}  # league-average priors
    recent = past.tail(ROLLING_WINDOW)
    form_recent = past.tail(FORM_WINDOW)
    return {
        "form": float(form_recent["pts"].mean()),
        "gf": float(recent["gf"].mean()),
        "ga": float(recent["ga"].mean()),
    }


def _h2h_bias(df: pd.DataFrame, home: str, away: str, before_date) -> float:
    """
    Head-to-head bias from the home team's perspective over prior meetings:
    (home wins - away wins) / meetings, in [-1, 1]. 0 if never met.
    """
    prior = df[
        (df["date"] < before_date)
        & (
            ((df["home_team"] == home) & (df["away_team"] == away))
            | ((df["home_team"] == away) & (df["away_team"] == home))
        )
    ]
    if prior.empty:
        return 0.0
    home_wins = away_wins = 0
    for _, r in prior.iterrows():
        if r["result"] == "D":
            continue
        winner = r["home_team"] if r["result"] == "H" else r["away_team"]
        if winner == home:
            home_wins += 1
        else:
            away_wins += 1
    return (home_wins - away_wins) / len(prior)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def _team_arrays(df: pd.DataFrame) -> dict[str, dict]:
    """
    Per-team chronological numpy arrays (pts, gf, ga, is_home). Precomputing
    these lets ``build_training_frame`` read pre-match windows with cheap array
    slices instead of a DataFrame filter per match — turning an O(n²) build
    into a near-linear one, which matters when training 22 leagues at startup.
    """
    import numpy as np

    hist = _team_match_history(df)
    out: dict[str, dict] = {}
    for team, h in hist.items():
        out[team] = {
            "pts": h["pts"].to_numpy(),
            "gf": h["gf"].to_numpy(dtype=float),
            "ga": h["ga"].to_numpy(dtype=float),
        }
    return out


def _fast_stats(arr: dict, before_idx: int) -> tuple[float, float, float]:
    """Form / gf / ga from a team's arrays using only matches before ``before_idx``."""
    if before_idx <= 0:
        return 1.0, 1.2, 1.2  # league-average priors
    form_win = arr["pts"][max(0, before_idx - FORM_WINDOW):before_idx]
    gf_win = arr["gf"][max(0, before_idx - ROLLING_WINDOW):before_idx]
    ga_win = arr["ga"][max(0, before_idx - ROLLING_WINDOW):before_idx]
    return float(form_win.mean()), float(gf_win.mean()), float(ga_win.mean())


def build_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert historical matches into a feature matrix + target column ``result``.
    Each row's features use only information available before that match.

    Note: ``home_goals_for``/``against`` here use a team's *overall* recent
    goal windows rather than home/away-split ones. On two seasons that keeps
    plenty of samples per window and makes the linear build fast; the home/away
    split is preserved in the (low-volume) inference path.
    """
    df = df.sort_values("date").reset_index(drop=True)
    arrays = _team_arrays(df)
    seen: dict[str, int] = {t: 0 for t in arrays}
    h2h_hist: dict[frozenset, list] = {}

    rows = []
    for h, a, res in zip(df["home_team"], df["away_team"], df["result"]):
        if h not in arrays or a not in arrays:
            continue

        home_form, home_gf, home_ga = _fast_stats(arrays[h], seen[h])
        away_form, away_gf, away_ga = _fast_stats(arrays[a], seen[a])

        # Incremental head-to-head bias from prior meetings (home perspective).
        key = frozenset((h, a))
        prior = h2h_hist.get(key, [])
        if prior:
            hw = sum(1 for w in prior if w == h)
            aw = sum(1 for w in prior if w == a)
            h2h_val = (hw - aw) / len(prior)
        else:
            h2h_val = 0.0

        rows.append(
            {
                "recent_form": home_form - away_form,
                "h2h": h2h_val,
                "home_goals_for": home_gf,
                "home_goals_against": home_ga,
                "away_goals_for": away_gf,
                "away_goals_against": away_ga,
                "result": res,
            }
        )

        # Record this match for future h2h lookups.
        winner = h if res == "H" else (a if res == "A" else None)
        h2h_hist.setdefault(key, []).append(winner)
        seen[h] += 1
        seen[a] += 1

    return pd.DataFrame(rows)


def team_strength_table(df: pd.DataFrame, last_n: int = 10) -> pd.DataFrame:
    """
    Model-free power ranking for a league, used when no live fixture feed is
    available. For each team, compute over its most recent ``last_n`` matches:
    form (points/game), goals for/against per game and goal difference.
    Sorted strongest-first. Empty frame if there is no data.
    """
    cols = ["team", "played", "form", "gf_per_game", "ga_per_game", "goal_diff"]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)

    # Use only the most recent season so a division's ranking reflects the teams
    # that actually play in it now — otherwise promoted/relegated sides show up
    # in last season's division (e.g. Wrexham lingering in League Two).
    if "season" in df.columns and df["season"].notna().any():
        latest = sorted(df["season"].dropna().unique())[-1]
        df = df[df["season"] == latest]

    team_hist = _team_match_history(df)
    rows = []
    for team, hist in team_hist.items():
        recent = hist.tail(last_n)
        if recent.empty:
            continue
        played = len(recent)
        gf = float(recent["gf"].mean())
        ga = float(recent["ga"].mean())
        rows.append(
            {
                "team": team,
                "played": played,
                "form": round(float(recent["pts"].mean()), 2),
                "gf_per_game": round(gf, 2),
                "ga_per_game": round(ga, 2),
                "goal_diff": round(gf - ga, 2),
            }
        )
    table = pd.DataFrame(rows, columns=cols)
    if table.empty:
        return table
    return table.sort_values(["form", "goal_diff"], ascending=False).reset_index(drop=True)


def build_fixture_features(df: pd.DataFrame, home: str, away: str) -> pd.DataFrame:
    """
    Build a single-row feature frame for an upcoming ``home`` vs ``away``
    fixture using all history in ``df``. Unknown teams fall back to priors,
    so predictions degrade gracefully instead of crashing.
    """
    df = df.sort_values("date").reset_index(drop=True)
    team_hist = _team_match_history(df)

    def stats(team: str, home_only: bool | None) -> dict:
        if team not in team_hist:
            return {"form": 1.0, "gf": 1.2, "ga": 1.2}
        return _pre_match_stats(team_hist[team], len(team_hist[team]), home_only)

    home_all, away_all = stats(home, None), stats(away, None)
    home_at_home = stats(home, True)
    away_at_away = stats(away, False)
    last_date = df["date"].max() if not df.empty else pd.Timestamp.now()

    row = {
        "recent_form": home_all["form"] - away_all["form"],
        "h2h": _h2h_bias(df, home, away, last_date + pd.Timedelta(days=1)),
        "home_goals_for": home_at_home["gf"],
        "home_goals_against": home_at_home["ga"],
        "away_goals_for": away_at_away["gf"],
        "away_goals_against": away_at_away["ga"],
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)
