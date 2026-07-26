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
        # Price-change signals
        "transfers_in_event", "transfers_out_event", "net_transfers_event",
        "cost_change_event",
        # Expected-stats (official FPL xG/xA)
        "xg", "xa", "xgi", "xgi_per_90",
        # Set-piece / penalty order (1 = first choice)
        "pens_order", "fk_order", "corners_order",
    ]
    if not bootstrap or "elements" not in bootstrap:
        return pd.DataFrame(columns=cols)

    teams = {t["id"]: t["name"] for t in bootstrap.get("teams", [])}
    rows = []
    for e in bootstrap["elements"]:
        tin = e.get("transfers_in_event", 0) or 0
        tout = e.get("transfers_out_event", 0) or 0
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
                "transfers_in_event": tin,
                "transfers_out_event": tout,
                "net_transfers_event": tin - tout,
                "cost_change_event": e.get("cost_change_event", 0) / 10.0,
                "xg": _to_float(e.get("expected_goals")),
                "xa": _to_float(e.get("expected_assists")),
                "xgi": _to_float(e.get("expected_goal_involvements")),
                "xgi_per_90": _to_float(e.get("expected_goal_involvements_per_90")),
                "pens_order": e.get("penalties_order"),
                "fk_order": e.get("direct_freekicks_order"),
                "corners_order": e.get("corners_and_indirect_freekicks_order"),
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
    win_prob: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Rank players for the captaincy. Score blends:
        - FPL expected points next GW (ep_next)
        - recent form
        - an ownership differential bonus (reward sub-30% owned differentials)
        - a fixture-ease bonus (easier upcoming fixtures score higher)
        - (optional) our own model's win probability for the player's team,
          via ``win_prob`` {team_name: P(team wins next match) in 0..1} — this
          links the match-outcome model to the captain pick.

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
    # Model win probability for the player's team (0.33 neutral prior if unknown).
    wp = win_prob or {}
    players["model_win_prob"] = players["team_name"].map(
        lambda t: round(wp.get(t, 0.33), 3)
    )
    players["captain_score"] = (
        players["ep_next"] * 1.0
        + players["form"] * 0.5
        + players["fixture_ease"] * 0.8
        + players["ownership_diff"] * 1.5
        + players["model_win_prob"] * 2.0  # our model's edge
    ).round(2)

    ranked = players.sort_values("captain_score", ascending=False).head(top_n)
    return ranked[
        [
            "web_name", "team_name", "position", "now_cost",
            "selected_by_percent", "form", "ep_next", "fixture_ease",
            "ownership_diff", "model_win_prob", "captain_score",
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


def price_change_watch(bootstrap: dict, top_n: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Predict likely overnight price rises/falls from net transfers this gameweek.

    FPL keeps the exact rise/fall thresholds hidden, so this ranks by net
    transfers (a strong proxy) rather than claiming certainty. ``cost_change``
    shows any change that has ALREADY happened this gameweek.

    Returns ``(risers, fallers)`` DataFrames.
    """
    players = players_frame(bootstrap)
    empty = pd.DataFrame(
        columns=["web_name", "team_name", "position", "now_cost",
                 "net_transfers_event", "cost_change_event"]
    )
    if players.empty:
        return empty, empty

    view = players[
        ["web_name", "team_name", "position", "now_cost",
         "net_transfers_event", "cost_change_event"]
    ]
    risers = view.sort_values("net_transfers_event", ascending=False).head(top_n)
    fallers = view.sort_values("net_transfers_event", ascending=True).head(top_n)
    # Only show fallers that are actually bleeding transfers.
    fallers = fallers[fallers["net_transfers_event"] < 0]
    return risers.reset_index(drop=True), fallers.reset_index(drop=True)


def transfer_targets(
    bootstrap: dict, fixtures: list[dict], weeks: int = 6, top_n: int = 5
) -> pd.DataFrame:
    """
    Best-value transfer targets per position, blending expected points, form,
    upcoming fixture ease and price. Returns one DataFrame with the top ``top_n``
    per position, plus a value metric (xPts per £m).
    """
    players = players_frame(bootstrap)
    cols = ["web_name", "team_name", "position", "now_cost", "form",
            "ep_next", "selected_by_percent", "value", "fixture_ease"]
    if players.empty:
        return pd.DataFrame(columns=cols)

    fdr = fixture_difficulty(bootstrap, fixtures, weeks)
    ease = {
        r["team_name"]: (5.0 - r["avg_difficulty"])
        for _, r in fdr.iterrows() if r["avg_difficulty"] is not None
    }
    avail = players[players["status"] == "a"].copy()
    if avail.empty:
        return pd.DataFrame(columns=cols)

    avail["fixture_ease"] = avail["team_name"].map(ease).fillna(2.0)
    avail["value"] = (avail["ep_next"] / avail["now_cost"].replace(0, 1)).round(2)
    avail["rank_score"] = (
        avail["ep_next"] + avail["form"] * 0.5 + avail["fixture_ease"] * 0.6
    )

    picks = []
    for pos in ["GK", "DEF", "MID", "FWD"]:
        pool = avail[avail["position"] == pos].sort_values("rank_score", ascending=False)
        picks.append(pool.head(top_n))
    out = pd.concat(picks) if picks else avail.head(0)
    return out[cols].reset_index(drop=True)


def fixture_ticker(bootstrap: dict, fixtures: list[dict], weeks: int = 8) -> pd.DataFrame:
    """
    A team × gameweek grid of fixture difficulty (the classic FPL ticker).
    Rows are teams, columns are ``GWn`` labels, cells hold the FDR (1 easy –
    5 hard) with the opponent, e.g. "3 CHE (H)". Empty frame on missing data.
    """
    if not bootstrap or not fixtures:
        return pd.DataFrame()

    teams = {t["id"]: t["name"] for t in bootstrap.get("teams", [])}
    short = {t["id"]: t.get("short_name", t["name"][:3].upper())
             for t in bootstrap.get("teams", [])}
    events = bootstrap.get("events", [])
    current = next((e["id"] for e in events if e.get("is_current")), None)
    if current is None:
        current = next((e["id"] for e in events if e.get("is_next")), 1)
    gws = list(range(current, current + weeks))

    grid: dict[int, dict[str, str]] = {tid: {} for tid in teams}
    for fx in fixtures:
        gw = fx.get("event")
        if gw not in gws:
            continue
        h, a = fx.get("team_h"), fx.get("team_a")
        hd = fx.get("team_h_difficulty", 3)
        ad = fx.get("team_a_difficulty", 3)
        col = f"GW{gw}"
        if h in grid:
            cell = f"{hd} {short.get(a, '?')} (H)"
            grid[h][col] = (grid[h].get(col) + " / " + cell) if grid[h].get(col) else cell
        if a in grid:
            cell = f"{ad} {short.get(h, '?')} (A)"
            grid[a][col] = (grid[a].get(col) + " / " + cell) if grid[a].get(col) else cell

    cols = [f"GW{g}" for g in gws]
    rows = []
    for tid, name in teams.items():
        row = {"Team": name}
        for c in cols:
            row[c] = grid[tid].get(c, "—")  # blank gameweek
        rows.append(row)
    return pd.DataFrame(rows).sort_values("Team").reset_index(drop=True)


def xg_leaders(bootstrap: dict, top_n: int = 10, squad_ids: set[int] | None = None) -> pd.DataFrame:
    """
    Players ranked by official FPL expected goal involvements (xG + xA).
    Restricted to a squad when ``squad_ids`` is given. Season totals — these are
    zero until matches are played.
    """
    players = players_frame(bootstrap)
    cols = ["web_name", "team_name", "position", "xg", "xa", "xgi", "xgi_per_90"]
    if players.empty:
        return pd.DataFrame(columns=cols)
    if squad_ids:
        players = players[players["id"].isin(squad_ids)]
    ranked = players.sort_values("xgi", ascending=False).head(top_n)
    return ranked[cols].reset_index(drop=True)


def personal_specials(specials: pd.DataFrame, bootstrap: dict, squad_ids: set[int] | None):
    """
    Given the league-wide special_gameweeks() frame, annotate each row with how
    many of the manager's players are affected. Returns the frame unchanged if
    no squad is set. Adds a ``your_double_players`` column.
    """
    if specials is None or specials.empty or not squad_ids:
        return specials
    players = players_frame(bootstrap)
    squad = players[players["id"].isin(squad_ids)]
    squad_teams = set(squad["team_name"])
    name_by_team = squad.groupby("team_name")["web_name"].apply(list).to_dict()

    def mine(cell: str) -> str:
        if not cell or cell == "—":
            return "—"
        teams_in_cell = [t.strip() for t in cell.split(",")]
        hit = [name_by_team[t] for t in teams_in_cell if t in squad_teams]
        flat = [p for sub in hit for p in sub]
        return ", ".join(flat) if flat else "—"

    out = specials.copy()
    out["your_double_players"] = out["double_teams"].apply(mine)
    return out


def chip_hints(specials: pd.DataFrame) -> list[str]:
    """
    Turn detected Double/Blank gameweeks into plain-English chip suggestions
    (Bench Boost, Triple Captain, Free Hit). Empty list if nothing notable.
    """
    if specials is None or specials.empty:
        return []
    hints = []
    for _, r in specials.iterrows():
        gw = r["gameweek"]
        if r["triple_teams"] != "—":
            hints.append(f"GW{gw}: Triple Gameweek — prime **Triple Captain** week ({r['triple_teams']}).")
        elif r["double_teams"] != "—":
            hints.append(f"GW{gw}: Double Gameweek — consider **Bench Boost** / **Triple Captain**.")
        if r["blank_teams"] != "—":
            hints.append(f"GW{gw}: Blank Gameweek — a **Free Hit** or bench cover may be needed.")
    return hints


def set_piece_takers(bootstrap: dict, squad_ids: set[int] | None = None) -> pd.DataFrame:
    """
    First-choice penalty / free-kick / corner takers (order == 1), the biggest
    hidden points edge in FPL. Restricted to a squad when ``squad_ids`` given.
    Columns: web_name, team_name, position, duties (e.g. "Pens, FKs").
    """
    players = players_frame(bootstrap)
    cols = ["web_name", "team_name", "position", "duties"]
    if players.empty:
        return pd.DataFrame(columns=cols)
    if squad_ids:
        players = players[players["id"].isin(squad_ids)]

    rows = []
    for _, p in players.iterrows():
        duties = []
        if p["pens_order"] == 1:
            duties.append("Pens")
        if p["fk_order"] == 1:
            duties.append("FKs")
        if p["corners_order"] == 1:
            duties.append("Corners")
        if duties:
            rows.append(
                {
                    "web_name": p["web_name"],
                    "team_name": p["team_name"],
                    "position": p["position"],
                    "duties": ", ".join(duties),
                }
            )
    df = pd.DataFrame(rows, columns=cols)
    # Penalty takers first, then by team.
    if not df.empty:
        df["_pen"] = df["duties"].str.contains("Pens").astype(int)
        df = df.sort_values(["_pen", "team_name"], ascending=[False, True]).drop(columns="_pen")
    return df.reset_index(drop=True)


def optimal_squad(bootstrap: dict, fixtures: list[dict], budget: float = 100.0, weeks: int = 6) -> dict:
    """
    Build a strong, valid FPL squad within ``budget`` (£m):
    2 GK, 5 DEF, 5 MID, 3 FWD, max 3 per club.

    Strategy: buy the cheapest legal 15-man squad first (guarantees feasibility),
    then hill-climb — repeatedly apply the single affordable same-position swap
    with the best projection gain until none remains. This reliably fills all
    15 slots and spends the budget well without a heavy LP solver.

    Returns ``{"squad": DataFrame, "total_cost", "total_ep", "feasible"}``.
    """
    players = players_frame(bootstrap)
    result = {"squad": pd.DataFrame(), "total_cost": 0.0, "total_ep": 0.0, "feasible": False}
    if players.empty:
        return result

    fdr = fixture_difficulty(bootstrap, fixtures, weeks)
    ease = {
        r["team_name"]: (5.0 - r["avg_difficulty"])
        for _, r in fdr.iterrows() if r["avg_difficulty"] is not None
    }
    pool = players[players["status"] == "a"].copy()
    if pool.empty:
        return result
    pool["fixture_ease"] = pool["team_name"].map(ease).fillna(2.0)
    pool["proj"] = pool["ep_next"] + pool["form"] * 0.5 + pool["fixture_ease"] * 0.6

    quotas = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
    # Per-position candidate lists as lightweight dicts.
    by_pos: dict[str, list[dict]] = {}
    for pos in quotas:
        by_pos[pos] = (
            pool[pool["position"] == pos]
            [["id", "web_name", "team_name", "position", "now_cost", "proj",
              "ep_next", "form", "fixture_ease"]]
            .to_dict("records")
        )
        if len(by_pos[pos]) < quotas[pos]:
            return result  # not enough players to form a legal squad

    picked: dict[int, dict] = {}
    club_count: dict[str, int] = {}

    def can_add(p, exclude_club: str | None = None) -> bool:
        c = club_count.get(p["team_name"], 0)
        if exclude_club == p["team_name"]:
            c -= 1  # a swap frees a slot at the same club
        return c < 3

    # Step 1: cheapest legal squad (feasibility base).
    for pos, need in quotas.items():
        cands = sorted(by_pos[pos], key=lambda x: x["now_cost"])
        taken = 0
        for p in cands:
            if taken >= need:
                break
            if p["id"] in picked or not can_add(p):
                continue
            picked[p["id"]] = p
            club_count[p["team_name"]] = club_count.get(p["team_name"], 0) + 1
            taken += 1
        if taken < need:
            return result  # club constraints made even the cheapest infeasible

    spent = sum(p["now_cost"] for p in picked.values())

    # Step 2: hill-climb upgrades within remaining budget.
    for _ in range(500):  # generous cap; converges well before this
        best_gain, best_swap = 0.0, None
        for out_p in list(picked.values()):
            for in_p in by_pos[out_p["position"]]:
                if in_p["id"] in picked:
                    continue
                gain = in_p["proj"] - out_p["proj"]
                if gain <= best_gain:
                    continue
                extra = in_p["now_cost"] - out_p["now_cost"]
                if spent + extra > budget:
                    continue
                if not can_add(in_p, exclude_club=out_p["team_name"]):
                    continue
                best_gain, best_swap = gain, (out_p, in_p)
        if not best_swap:
            break
        out_p, in_p = best_swap
        del picked[out_p["id"]]
        club_count[out_p["team_name"]] -= 1
        picked[in_p["id"]] = in_p
        club_count[in_p["team_name"]] = club_count.get(in_p["team_name"], 0) + 1
        spent += in_p["now_cost"] - out_p["now_cost"]

    squad = pd.DataFrame(list(picked.values()))
    order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    squad["_o"] = squad["position"].map(order)
    squad = squad.sort_values(["_o", "ep_next"], ascending=[True, False]).drop(columns="_o")
    squad = squad[["web_name", "team_name", "position", "now_cost", "ep_next", "form", "fixture_ease"]]

    result["squad"] = squad.reset_index(drop=True)
    result["total_cost"] = round(spent, 1)
    result["total_ep"] = round(float(squad["ep_next"].sum()), 1)
    result["feasible"] = len(picked) == 15
    return result


def transfer_suggestions(
    bootstrap: dict,
    fixtures: list[dict],
    squad_ids: set[int],
    weeks: int = 6,
    n: int = 5,
) -> pd.DataFrame:
    """
    Compare the manager's actual squad to the best available alternatives and
    recommend the transfers that most improve projected points.

    For each owned player, find the best same-position player you DON'T own
    (respecting the max-3-per-club rule given the rest of your squad), then rank
    all such swaps by projection gain. Shows the price delta so you can judge
    affordability (we can't see your bank balance).

    Columns: out_player, out_team, in_player, in_team, position, gain, cost_delta.
    Empty frame if no squad is set or no improving swap exists.
    """
    cols = ["out_player", "out_team", "in_player", "in_team",
            "position", "gain", "cost_delta"]
    players = players_frame(bootstrap)
    if players.empty or not squad_ids:
        return pd.DataFrame(columns=cols)

    fdr = fixture_difficulty(bootstrap, fixtures, weeks)
    ease = {
        r["team_name"]: (5.0 - r["avg_difficulty"])
        for _, r in fdr.iterrows() if r["avg_difficulty"] is not None
    }
    players = players.copy()
    players["fixture_ease"] = players["team_name"].map(ease).fillna(2.0)
    players["proj"] = players["ep_next"] + players["form"] * 0.5 + players["fixture_ease"] * 0.6

    squad = players[players["id"].isin(squad_ids)]
    if squad.empty:
        return pd.DataFrame(columns=cols)
    # Club counts in the current squad (for the 3-per-club rule).
    club_count = squad["team_name"].value_counts().to_dict()
    available = players[(~players["id"].isin(squad_ids)) & (players["status"] == "a")]

    swaps = []
    for _, out_p in squad.iterrows():
        pos = out_p["position"]
        cands = available[available["position"] == pos].sort_values("proj", ascending=False)
        for _, in_p in cands.iterrows():
            if in_p["proj"] <= out_p["proj"]:
                break  # sorted desc, nothing better remains
            # Club rule: adding in_p removes out_p first.
            eff = club_count.get(in_p["team_name"], 0)
            if in_p["team_name"] == out_p["team_name"]:
                eff -= 1
            if eff >= 3:
                continue
            swaps.append(
                {
                    "out_player": out_p["web_name"],
                    "out_team": out_p["team_name"],
                    "in_player": in_p["web_name"],
                    "in_team": in_p["team_name"],
                    "position": pos,
                    "gain": round(float(in_p["proj"] - out_p["proj"]), 2),
                    "cost_delta": round(float(in_p["now_cost"] - out_p["now_cost"]), 1),
                }
            )
            break  # best candidate for this player found

    df = pd.DataFrame(swaps, columns=cols)
    if df.empty:
        return df
    # Best-gain first, then keep each incoming/outgoing player only once — you
    # can't sign the same target for several slots or sell a player twice.
    df = df.sort_values("gain", ascending=False)
    df = df.drop_duplicates(subset="in_player").drop_duplicates(subset="out_player")
    return df.head(n).reset_index(drop=True)


def squad_ids_from_picks(picks_payload: dict) -> set[int]:
    """Extract element (player) ids from an FPL entry/picks payload."""
    if not picks_payload or "picks" not in picks_payload:
        return set()
    return {p["element"] for p in picks_payload["picks"] if "element" in p}
