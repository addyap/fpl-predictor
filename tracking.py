"""
tracking.py
===========
Lightweight, self-contained persistence for a live prediction track record.

Each time you log a gameweek, its predictions are stored in a local SQLite file.
When results come in, logged rows are scored so the app can show real accuracy
over time — a *live* track record to complement the historical backtest.

IMPORTANT: storage is a local file (``data/predictions.db``). On a persistent
host that survives forever. On Streamlit Community Cloud the free tier's disk is
ephemeral, so the log resets whenever the app reboots/redeploys — point
``FPL_DB_PATH`` (or the Supabase hook below) at durable storage for permanence.

Every function is wrapped defensively: if the DB can't be opened the app keeps
working, tracking simply reports "unavailable" instead of crashing.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager

import pandas as pd

DB_PATH = os.environ.get(
    "FPL_DB_PATH", os.path.join(os.path.dirname(__file__), "data", "predictions.db")
)


@contextmanager
def _conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db() -> bool:
    """Create the predictions table if needed. Returns False if storage is unusable."""
    try:
        with _conn() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    gw INTEGER,
                    home TEXT,
                    away TEXT,
                    pred TEXT,             -- predicted result letter H/D/A
                    prob_home REAL,
                    prob_draw REAL,
                    prob_away REAL,
                    actual TEXT,           -- H/D/A once known, else NULL
                    correct INTEGER,       -- 1/0 once scored, else NULL
                    UNIQUE(gw, home, away)
                )
                """
            )
        return True
    except sqlite3.Error as exc:
        print(f"[tracking] init failed: {exc}")
        return False


def log_predictions(gw: int, records: list[dict]) -> int:
    """
    Store predictions for a gameweek. ``records`` are dicts with keys
    home, away, pred, prob_home, prob_draw, prob_away. Duplicate (gw, home,
    away) rows are ignored, so logging twice is safe. Returns rows inserted.
    """
    if not records or not init_db():
        return 0
    try:
        with _conn() as con:
            cur = con.executemany(
                """
                INSERT OR IGNORE INTO predictions
                    (gw, home, away, pred, prob_home, prob_draw, prob_away)
                VALUES (:gw, :home, :away, :pred, :prob_home, :prob_draw, :prob_away)
                """,
                [{**r, "gw": gw} for r in records],
            )
            return cur.rowcount or 0
    except sqlite3.Error as exc:
        print(f"[tracking] log failed: {exc}")
        return 0


def update_results(results: pd.DataFrame) -> int:
    """
    Score logged predictions against finished results. ``results`` needs columns
    home_team, away_team, result (H/D/A). Matches on exact home/away names.
    Returns the number of rows newly scored.
    """
    if results is None or results.empty or not init_db():
        return 0
    scored = 0
    try:
        with _conn() as con:
            for _, r in results.iterrows():
                cur = con.execute(
                    """
                    UPDATE predictions
                       SET actual = ?, correct = CASE WHEN pred = ? THEN 1 ELSE 0 END
                     WHERE home = ? AND away = ? AND actual IS NULL
                    """,
                    (r["result"], r["result"], r["home_team"], r["away_team"]),
                )
                scored += cur.rowcount or 0
        return scored
    except sqlite3.Error as exc:
        print(f"[tracking] update failed: {exc}")
        return 0


def get_logged() -> pd.DataFrame:
    """All logged predictions (scored and pending). Empty frame if unavailable."""
    cols = ["gw", "home", "away", "pred", "prob_home", "prob_draw",
            "prob_away", "actual", "correct"]
    if not init_db():
        return pd.DataFrame(columns=cols)
    try:
        with _conn() as con:
            return pd.read_sql_query("SELECT * FROM predictions ORDER BY gw", con)
    except (sqlite3.Error, pd.errors.DatabaseError) as exc:
        print(f"[tracking] read failed: {exc}")
        return pd.DataFrame(columns=cols)


def summary() -> dict:
    """
    Track-record summary: total logged, total scored, overall accuracy, and a
    per-gameweek accuracy DataFrame for charting. Safe defaults if empty.
    """
    df = get_logged()
    out = {"n_logged": 0, "n_scored": 0, "accuracy": None, "by_gw": pd.DataFrame()}
    if df.empty:
        return out
    out["n_logged"] = len(df)
    scored = df[df["correct"].notna()]
    out["n_scored"] = len(scored)
    if not scored.empty:
        out["accuracy"] = round(float(scored["correct"].mean()), 3)
        by_gw = (
            scored.groupby("gw")["correct"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "accuracy", "count": "matches"})
        )
        by_gw["accuracy"] = (by_gw["accuracy"] * 100).round(1)
        out["by_gw"] = by_gw
    return out
