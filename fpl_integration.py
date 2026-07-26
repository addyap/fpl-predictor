"""
fpl_integration.py
==================
Bridges the match-outcome model to Fantasy Premier League decisions:

- Normalises the FPL bootstrap into a tidy players DataFrame
- Computes fixture-difficulty over the next N gameweeks (radar data)
- Produces a captain shortlist ranked by form, expected points and an
  ownership differential (rewarding effective, lower-owned picks)
- Surfaces injury / availability alerts
- Personalises everything to a manager's squad when a team id is supplied

Every function tolerates empty inputs (returns empty frames / neutral values)
so the UI can render "N/A" instead of crashing when data is missing.
"""

from __future__ import annotations

import pandas as pd

# Map FPL element_type ids to readable positions.
POSITIONS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

# FPL 'status' codes -> human availability. 'a' == available.
STATUS = {
    "a": "Available",
    "d": "Doubtful",
    "i": "Injured",
    "s": "Suspended",
    "u": "Unavailable",
    "n": "Not in squad",
}


# --------------------------------------------------------------------------- #
# Bootstrap normalisation
# --------------------------------------------------------------------------- #

def players_frame(bootstrap: dict) -> pd.DataFrame:
    """
    Flatten bootstrap-static 'elements' into a DataFrame with the columns the
    rest of the module relies on. Empty frame if bootstrap is missing.
    """
    cols = [
        "id", "web_name", "team", "team_name", "position", "now_cost",
        "selected_by_percent", "form", "total_points", "points_per_game",
        "status", "news", "ep_next",
    ]
    if not bootstrap or "elements" not in bootstrap:
        return pd.DataFrame(columns=cols)

    teams = {t["id"]: t["name"] for t in bootstrap.get("teams", [])}
    rows = []
    for e in bootstrap["elements"]:
        rows.append(
            {
                "id": e.get("id"),
                "web_name": e.get("web_name", "N/A"),
                "team": e.get("team"),
                "team_name": teams.get(e.get("team"), "N/A"),
                "position": POSITIONS.get(e.get("element_type"), "N/A"),
                "now_cost": e.get("now_cost", 0) / 10.0,  # tenths of a million
                "selected_by_percent": _to_float(e.get("selected_by_percent")),
                "form": _to_float(e.get("form")),
                "total_points": e.get("total_points", 0),
                "points_per_game": _to_float(e.get("points_per_game")),
                "status": e.get("status", "a"),
                "news": e.get("news", ""),
                "ep_next": _to_float(e.get("ep_next")),  # FPL's own expected pts
            }
        )
    return pd.DataFrame(rows, columns=cols)


def _to_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


# --------------------------------------------------------------------------- #
# Fixture difficulty
# --------------------------------------------------------------------------- #

def fixture_difficulty(
    bootstrap: dict, fixtures: list[dict], weeks: int = 8
) -> pd.DataFrame:
    """
    Average FPL fixture-difficulty rating (FDR, 1=easy … 5=hard) per team over
    the next ``weeks`` gameweeks. Columns: team_name, avg_difficulty, n_games.
    """
    cols = ["team_name", "avg_difficulty", "n_games"]
    if not bootstrap or not fixtures:
        return pd.DataFrame(columns=cols)

    teams = {t["id"]: t["name"] for t in bootstrap.get("teams", [])}
    events = bootstrap.get("events", [])
    current = next((e["id"] for e in events if e.get("is_current")), None)
    if current is None:
        current = next((e["id"] for e in events if e.get("is_next")), 1)
    horizon = set(range(current, current + weeks))

    agg: dict[int, list[int]] = {}
    for fx in fixtures:
        gw = fx.get("event")
        if gw not in horizon:
            continue
        h, a = fx.get("team_h"), fx.get("team_a")
        agg.setdefault(h, []).append(fx.get("team_h_difficulty", 3))
        agg.setdefault(a, []).append(fx.get("team_a_difficulty", 3))

    rows = [
        {
            "team_name": teams.get(tid, "N/A"),
            "avg_difficulty": round(sum(d) / len(d), 2) if d else None,
            "n_games": len(d),
        }
        for tid, d in agg.items()
    ]
    return pd.DataFrame(rows, columns=cols).sort_values("avg_difficulty").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Captain / transfer recommendations
# --------------------------------------------------------------------------- #

def captain_recommendations(
    bootstrap: dict,
    fixtures: list[dict],
    weeks: int = 8,
    top_n: int = 8,
    squad_ids: set[int] | None = None,
) -> pd.DataFrame:
    """
    Rank players for the captaincy. Score blends:
        - FPL expected points next GW (ep_next)
        - recent form
        - an ownership differential bonus (reward sub-30% owned differentials)
        - a fixture-ease bonus (easier upcoming fixtures score higher)

    If ``squad_ids`` is given, only those players are considered (personalised);
    otherwise the whole player pool is used (generic recommendations).
    """
    players = players_frame(bootstrap)
    if players.empty:
        return players

    if squad_ids:
        players = players[players["id"].isin(squad_ids)].copy()
        if players.empty:
            return players

    # Fixture ease per team over the horizon (5 - difficulty, so higher=easier).
    fdr = fixture_difficulty(bootstrap, fixtures, weeks)
    ease = {
        row["team_name"]: (5.0 - row["avg_difficulty"])
        for _, row in fdr.iterrows()
        if row["avg_difficulty"] is not None
    }

    players = players[players["status"] == "a"].copy()  # only available players
    if players.empty:
        return players

    def ownership_bonus(pct: float) -> float:
        # Reward differentials: full bonus under 10%, tapering to 0 by 50%.
        if pct >= 50:
            return 0.0
        return round((50 - pct) / 50, 3)

    players["fixture_ease"] = players["team_name"].map(ease).fillna(2.0)
    players["ownership_diff"] = players["selected_by_percent"].apply(ownership_bonus)
    players["captain_score"] = (
        players["ep_next"] * 1.0
        + players["form"] * 0.5
        + players["fixture_ease"] * 0.8
        + players["ownership_diff"] * 1.5
    ).round(2)

    ranked = players.sort_values("captain_score", ascending=False).head(top_n)
    return ranked[
        [
            "web_name", "team_name", "position", "now_cost",
            "selected_by_percent", "form", "ep_next", "fixture_ease",
            "ownership_diff", "captain_score",
        ]
    ].reset_index(drop=True)


def special_gameweeks(bootstrap: dict, fixtures: list[dict], ahead: int = 15) -> pd.DataFrame:
    """
    Detect Double/Triple Gameweeks (a team plays 2+/3+ times in one gameweek)
    and Blank Gameweeks (a team has no fixture while others play) over the next
    ``ahead`` gameweeks.

    Doubles/blanks arise when the Premier League reschedules matches around cup
    rounds, so early in a season the fixture list is usually all singles and
    this returns an empty frame — the specials get confirmed by the FPL API as
    the season progresses.

    Returns one row per notable gameweek, columns:
    ``gameweek``, ``type`` (Double/Triple/Blank/Double+Blank…),
    ``double_teams``, ``triple_teams``, ``blank_teams``.
    """
    cols = ["gameweek", "type", "double_teams", "triple_teams", "blank_teams"]
    if not bootstrap or not fixtures:
        return pd.DataFrame(columns=cols)

    teams = {t["id"]: t["name"] for t in bootstrap.get("teams", [])}
    events = bootstrap.get("events", [])
    current = next((e["id"] for e in events if e.get("is_current")), None)
    if current is None:
        current = next((e["id"] for e in events if e.get("is_next")), 1)
    horizon = set(range(current, current + ahead))

    # gameweek -> {team_id: fixture_count}
    counts: dict[int, dict[int, int]] = {}
    for fx in fixtures:
        gw = fx.get("event")
        if gw is None or gw not in horizon:
            continue
        bucket = counts.setdefault(gw, {})
        for side in ("team_h", "team_a"):
            tid = fx.get(side)
            if tid is not None:
                bucket[tid] = bucket.get(tid, 0) + 1

    rows = []
    for gw in sorted(counts):
        team_counts = counts[gw]
        if not team_counts:
            continue
        doubles = sorted(teams.get(t, "?") for t, c in team_counts.items() if c == 2)
        triples = sorted(teams.get(t, "?") for t, c in team_counts.items() if c >= 3)
        # A blank is a team with no fixture in a gameweek that others do play.
        playing = set(team_counts)
        blanks = sorted(teams[t] for t in teams if t not in playing)

        if not (doubles or triples or blanks):
            continue  # a normal all-singles gameweek

        labels = []
        if triples:
            labels.append("Triple")
        if doubles:
            labels.append("Double")
        if blanks:
            labels.append("Blank")
        rows.append(
            {
                "gameweek": gw,
                "type": " + ".join(labels) + " GW",
                "double_teams": ", ".join(doubles) if doubles else "—",
                "triple_teams": ", ".join(triples) if triples else "—",
                "blank_teams": ", ".join(blanks) if blanks else "—",
            }
        )
    return pd.DataFrame(rows, columns=cols)


def next_gameweek_info(bootstrap: dict) -> dict:
    """Return {'current': id, 'next': id, 'name': str} for display. Empty on failure."""
    events = bootstrap.get("events", []) if bootstrap else []
    if not events:
        return {}
    current = next((e for e in events if e.get("is_current")), None)
    nxt = next((e for e in events if e.get("is_next")), None)
    ref = current or nxt or events[0]
    return {
        "current": current["id"] if current else None,
        "next": nxt["id"] if nxt else None,
        "name": ref.get("name", "N/A"),
    }


def injury_alerts(bootstrap: dict, squad_ids: set[int] | None = None) -> pd.DataFrame:
    """
    Players flagged as not fully available. Restricted to the manager's squad
    when ``squad_ids`` is provided. Columns: web_name, team_name, availability, news.
    """
    players = players_frame(bootstrap)
    cols = ["web_name", "team_name", "availability", "news"]
    if players.empty:
        return pd.DataFrame(columns=cols)

    if squad_ids:
        players = players[players["id"].isin(squad_ids)]

    flagged = players[players["status"] != "a"].copy()
    if flagged.empty:
        return pd.DataFrame(columns=cols)

    flagged["availability"] = flagged["status"].map(STATUS).fillna("Unknown")
    return flagged[["web_name", "team_name", "availability", "news"]].reset_index(drop=True)


def squad_ids_from_picks(picks_payload: dict) -> set[int]:
    """Extract element (player) ids from an FPL entry/picks payload."""
    if not picks_payload or "picks" not in picks_payload:
        return set()
    return {p["element"] for p in picks_payload["picks"] if "element" in p}
