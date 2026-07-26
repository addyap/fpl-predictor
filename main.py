"""
main.py
=======
Streamlit UI for the FPL predictor.

Run with:
    streamlit run main.py

On first run it downloads two seasons of Premier League history (with a
progress bar), trains the logistic model once, then lets you refresh live FPL /
football-data.org data on demand across three tabs.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data_fetch
import features
import fpl_integration as fpl
import model as model_mod

# --------------------------------------------------------------------------- #
# Page config
# --------------------------------------------------------------------------- #

st.set_page_config(page_title="FPL Predictor", page_icon="⚽", layout="wide")

REFRESH_LEVELS = {
    "Quick (5s)": "quick",
    "Full (20s)": "full",
    "Heavy (2min)": "heavy",
}


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner=False)
def get_all_models():
    """
    Train one model per league once per session (cached). Assumes the
    historical CSVs already exist — the first-run download is handled
    separately so we can show a progress bar outside the cache.
    Returns ``(models, histories)`` dicts keyed by league code.
    """
    models: dict = {}
    histories: dict = {}
    for lg in data_fetch.LEAGUES:
        try:
            df = data_fetch.load_historical(lg["code"])
        except Exception as exc:  # league CSV missing/unreadable — skip it
            print(f"[main] could not load {lg['code']}: {exc}")
            continue
        histories[lg["code"]] = df
        models[lg["code"]] = model_mod.train_model(df)
    return models, histories


@st.cache_data(ttl=300, show_spinner=False)
def cached_bootstrap():
    return data_fetch.fetch_fpl_bootstrap()


@st.cache_data(ttl=300, show_spinner=False)
def cached_fixtures():
    return data_fetch.fetch_fpl_fixtures()


@st.cache_data(ttl=300, show_spinner=False)
def cached_upcoming(api_key: str, competition: str):
    return data_fetch.fetch_upcoming_matches(api_key, competition)


@st.cache_data(ttl=300, show_spinner=False)
def cached_recent_results(api_key: str, competition: str):
    return data_fetch.fetch_recent_results(api_key, competition)


@st.cache_data(ttl=300, show_spinner=False)
def cached_team(team_id: str):
    if not team_id:
        return {}
    return data_fetch.fetch_fpl_team(int(team_id)) if team_id.isdigit() else {}


# --------------------------------------------------------------------------- #
# First-run setup: download historical data with a progress bar
# --------------------------------------------------------------------------- #

def ensure_historical():
    if data_fetch.all_historical_exists():
        return True
    n = len(data_fetch.LEAGUES)
    st.info(f"First run: downloading {n} leagues × 2 seasons (2023/24 + 2024/25)…")
    bar = st.progress(0.0, text="Starting download…")

    def _cb(frac: float, msg: str):
        bar.progress(min(max(frac, 0.0), 1.0), text=msg)

    try:
        data_fetch.download_all_historical(progress=_cb)
    except RuntimeError as exc:
        bar.empty()
        st.error(str(exc))
        return False

    # Premier League is required for the first two tabs; everything else is
    # best-effort (a league with no upstream data is simply skipped).
    if not data_fetch.historical_exists(data_fetch.PL_CODE):
        bar.empty()
        st.error("Could not download Premier League history. Check your connection and reload.")
        return False

    bar.progress(1.0, text="Done.")
    bar.empty()
    st.success("Historical data downloaded for all available leagues.")
    return True


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #

def sidebar() -> dict:
    st.sidebar.header("⚙️ Configuration")

    default_key = st.secrets.get("football_data_org_key", "") if hasattr(st, "secrets") else ""
    api_key = st.sidebar.text_input(
        "football-data.org API key",
        value=default_key,
        type="password",
        help="Required for live fixtures & results.",
    )
    st.sidebar.caption(
        "Get a free key at [football-data.org](https://www.football-data.org/) "
        "(sign up takes 2 min)."
    )

    default_team = st.secrets.get("fpl_team_id", "") if hasattr(st, "secrets") else ""
    team_id = st.sidebar.text_input(
        "FPL team ID (optional)",
        value=str(default_team),
        help="The number in your FPL points-page URL. Leave blank for generic picks.",
    )

    level_label = st.sidebar.selectbox("Refresh level", list(REFRESH_LEVELS.keys()), index=0)

    st.sidebar.divider()
    st.sidebar.caption("⚠️ Not financial advice. Gamble responsibly.")

    return {
        "api_key": api_key.strip(),
        "team_id": team_id.strip(),
        "level": REFRESH_LEVELS[level_label],
    }


# --------------------------------------------------------------------------- #
# Refresh handling
# --------------------------------------------------------------------------- #

def do_refresh(level: str, api_key: str):
    """Clear the relevant caches so the next reads fetch fresh data."""
    cached_bootstrap.clear()
    cached_fixtures.clear()
    cached_upcoming.clear()
    cached_recent_results.clear()
    cached_team.clear()

    now = datetime.now().strftime("%I:%M%p").lstrip("0").lower()
    st.session_state["last_update"] = now
    st.session_state["last_level"] = level


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #

def _predictions_table(trained, history, upcoming) -> pd.DataFrame:
    """Build a display-ready predictions DataFrame from an upcoming-fixtures frame."""
    known = set(history["home_team"]) | set(history["away_team"])
    rows = []
    for _, fx in upcoming.iterrows():
        home = _match_team_name(fx["home_team"], known)
        away = _match_team_name(fx["away_team"], known)
        pred = model_mod.predict_fixture(trained, history, home, away)
        odds = _implied_odds_comparison(history, home, away, pred)
        rows.append(
            {
                "Match": f"{fx['home_team']} vs {fx['away_team']}",
                "Kickoff (UTC)": _fmt_date(fx["utc_date"]),
                "Predicted": pred["predicted"],
                "Confidence": f"{pred['confidence']}%",
                "Home %": pred["prob_home"],
                "Draw %": pred["prob_draw"],
                "Away %": pred["prob_away"],
                "Model odds (H/D/A)": odds["model"],
                "Hist. book odds": odds["book"],
            }
        )
    return pd.DataFrame(rows)


def tab_next_gameweek(trained, history, api_key):
    st.subheader("Next Gameweek — predicted outcomes")

    if not api_key:
        st.warning("Enter your football-data.org API key in the sidebar to load upcoming fixtures.")
        return

    upcoming = cached_upcoming(api_key, data_fetch.PL_COMPETITION)
    if upcoming.empty:
        st.info("No upcoming fixtures returned. Check your API key, or try again shortly.")
        return

    df = _predictions_table(trained, history, upcoming)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption(
        f"Model trained on {trained.n_matches} matches · "
        f"training accuracy {trained.accuracy:.1%}. "
        "Betting odds shown are for comparison only."
    )


def tab_fpl(trained, api_key, team_id):
    st.subheader("FPL Recommendations")
    boot = cached_bootstrap()
    fixtures = cached_fixtures()

    if not boot:
        st.error("Couldn't reach the FPL API. Try a refresh in a moment.")
        return

    squad_ids = None
    if team_id:
        picks = cached_team(team_id)
        squad_ids = fpl.squad_ids_from_picks(picks)
        if squad_ids:
            st.success(f"Personalised to team {team_id} ({len(squad_ids)} players).")
        else:
            st.warning(f"Couldn't load squad for team id '{team_id}'. Showing generic picks.")
    else:
        st.info("No FPL team id set — showing generic recommendations.")

    # Captain pick
    caps = fpl.captain_recommendations(boot, fixtures, weeks=8, top_n=8, squad_ids=squad_ids)
    st.markdown("### 🧢 Captain shortlist")
    if caps.empty:
        st.write("N/A — no eligible players found.")
    else:
        top = caps.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Top captain", f"{top['web_name']}", f"{top['team_name']}")
        c2.metric("Expected pts (next)", f"{top['ep_next']:.1f}")
        c3.metric("Ownership", f"{top['selected_by_percent']:.1f}%")
        st.dataframe(
            caps.rename(
                columns={
                    "web_name": "Player", "team_name": "Team", "position": "Pos",
                    "now_cost": "£m", "selected_by_percent": "Owned %",
                    "form": "Form", "ep_next": "xPts", "fixture_ease": "Fixture ease",
                    "ownership_diff": "Diff bonus", "captain_score": "Score",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    # Fixture difficulty radar
    st.markdown("### 🎯 Fixture difficulty (next 8 GWs)")
    fdr = fpl.fixture_difficulty(boot, fixtures, weeks=8)
    if fdr.empty:
        st.write("N/A — fixture data unavailable.")
    else:
        _render_fdr_radar(fdr, squad_ids, boot)

    # Injury alerts
    st.markdown("### 🚑 Injury / availability alerts")
    inj = fpl.injury_alerts(boot, squad_ids=squad_ids)
    if inj.empty:
        st.write("No availability concerns" + (" in your squad." if squad_ids else "."))
    else:
        st.dataframe(
            inj.rename(
                columns={
                    "web_name": "Player", "team_name": "Team",
                    "availability": "Status", "news": "News",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


def tab_all_leagues(models, histories, api_key):
    st.subheader("Predictions — All Leagues")
    live = [lg for lg in data_fetch.LEAGUES if lg["org"]]
    ranked_only = [lg for lg in data_fetch.LEAGUES if not lg["org"]]
    st.caption(
        f"{len(models)} leagues trained · {len(live)} with live upcoming fixtures "
        f"(football-data.org free tier) · {len(ranked_only)} shown as model-based "
        "power rankings (no live fixture feed on the free tier)."
    )

    if not api_key:
        st.warning(
            "No football-data.org key set — live fixtures are unavailable, so every "
            "league below falls back to a model-based power ranking. Add your key in "
            "the sidebar to unlock live predictions for the top leagues."
        )

    # Fetching live fixtures for ~8 leagues hits the API, so gate it behind a
    # button to keep the app snappy and stay under the free-tier rate limit.
    load_live = st.session_state.get("all_leagues_loaded", False)
    if api_key and not load_live:
        if st.button("🌍 Load live fixtures for all leagues", use_container_width=True):
            st.session_state["all_leagues_loaded"] = True
            st.rerun()
        st.info("Power rankings shown below now. Click above to also pull live fixture predictions.")

    for lg in data_fetch.LEAGUES:
        code = lg["code"]
        if code not in models:
            continue
        history = histories[code]
        trained = models[code]
        has_live = bool(lg["org"]) and api_key and load_live

        # Live-capable leagues open by default so the fixtures are "all visible".
        with st.expander(lg["name"], expanded=bool(lg["org"])):
            rendered_predictions = False
            if has_live:
                upcoming = cached_upcoming(api_key, lg["org"])
                if not upcoming.empty:
                    table = _predictions_table(trained, history, upcoming)
                    st.markdown("**Upcoming fixtures — predicted outcomes**")
                    st.dataframe(table, use_container_width=True, hide_index=True)
                    rendered_predictions = True
                else:
                    st.caption(
                        "No live fixtures returned (off-season or rate-limited) — "
                        "showing power ranking instead."
                    )

            # Always show the power ranking; for non-live leagues it's the main view.
            ranking = features.team_strength_table(history)
            st.markdown("**Team power ranking** (last 10 games)")
            if ranking.empty:
                st.write("N/A")
            else:
                st.dataframe(
                    ranking.rename(
                        columns={
                            "team": "Team", "played": "P", "form": "Form (pts/gm)",
                            "gf_per_game": "GF/gm", "ga_per_game": "GA/gm",
                            "goal_diff": "GD/gm",
                        }
                    ).head(12),
                    use_container_width=True,
                    hide_index=True,
                )
            st.caption(
                f"Model: {trained.n_matches} matches · train acc {trained.accuracy:.1%}"
                + ("" if rendered_predictions else " · live fixtures not available for this league")
            )


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _match_team_name(name: str, known: set[str]) -> str:
    """
    football-data.org names ('Arsenal FC') differ from football-data.co.uk
    ('Arsenal'). Do a light fuzzy match so predictions line up; fall back to
    the raw name (features.py handles unknown teams with priors).
    """
    if name in known:
        return name
    stripped = name.replace(" FC", "").replace(" AFC", "").strip()
    if stripped in known:
        return stripped
    for k in known:
        if k.lower() in name.lower() or name.lower() in k.lower():
            return k
    return stripped


def _fmt_date(utc_str: str) -> str:
    try:
        dt = datetime.fromisoformat(utc_str.replace("Z", "+00:00"))
        return dt.strftime("%a %d %b, %H:%M")
    except (ValueError, AttributeError):
        return "N/A"


def _implied_odds_comparison(history, home, away, pred) -> dict:
    """Model probabilities as decimal odds, plus historical Bet365 avg if any."""
    def dec(p):
        return round(100.0 / p, 2) if p and p > 0 else "N/A"

    model_odds = f"{dec(pred['prob_home'])} / {dec(pred['prob_draw'])} / {dec(pred['prob_away'])}"

    book = "N/A"
    if {"odds_home", "odds_draw", "odds_away"}.issubset(history.columns):
        past = history[(history["home_team"] == home) & (history["away_team"] == away)]
        if not past.empty:
            h = past["odds_home"].dropna().mean()
            d = past["odds_draw"].dropna().mean()
            a = past["odds_away"].dropna().mean()
            if pd.notna(h) and pd.notna(d) and pd.notna(a):
                book = f"{h:.2f} / {d:.2f} / {a:.2f}"
    return {"model": model_odds, "book": book}


def _render_fdr_radar(fdr: pd.DataFrame, squad_ids, boot):
    """Radar chart of fixture difficulty for up to 8 teams (squad teams first)."""
    teams_to_show = fdr.copy()
    if squad_ids:
        players = fpl.players_frame(boot)
        squad_teams = set(players[players["id"].isin(squad_ids)]["team_name"])
        prioritized = teams_to_show[teams_to_show["team_name"].isin(squad_teams)]
        teams_to_show = prioritized if not prioritized.empty else teams_to_show
    teams_to_show = teams_to_show.head(8)

    categories = teams_to_show["team_name"].tolist()
    values = teams_to_show["avg_difficulty"].tolist()
    if not categories:
        st.write("N/A")
        return

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values + values[:1],
            theta=categories + categories[:1],
            fill="toself",
            name="Avg difficulty (1 easy – 5 hard)",
            line_color="#e90052",
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
        showlegend=False,
        margin=dict(l=40, r=40, t=20, b=20),
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Lower = easier upcoming fixtures. Squad teams shown first when a team id is set.")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    st.title("⚽ FPL Predictor")
    st.caption("Premier League outcome model + Fantasy Premier League recommendations.")

    if not ensure_historical():
        st.stop()

    cfg = sidebar()

    with st.spinner("Loading / training models for all leagues (first run only)…"):
        models, histories = get_all_models()
    trained = models.get(data_fetch.PL_CODE)
    history = histories.get(data_fetch.PL_CODE)
    if trained is None or history is None:
        st.error("Premier League model unavailable. Try reloading the page.")
        st.stop()

    # Validate API key if provided
    if cfg["api_key"]:
        ok, msg = _validate_key_cached(cfg["api_key"])
        if not ok:
            st.sidebar.error(msg)
        else:
            st.sidebar.success(msg)

    # Refresh buttons
    b1, b2, b3 = st.columns(3)
    if b1.button("🔄 Quick Refresh", use_container_width=True):
        do_refresh("quick", cfg["api_key"])
    if b2.button("🔄 Full Refresh", use_container_width=True, help="Includes xG (placeholder)"):
        do_refresh("full", cfg["api_key"])
    if b3.button("🔄 Heavy Refresh", use_container_width=True, help="Includes injuries (placeholder)"):
        do_refresh("heavy", cfg["api_key"])

    last = st.session_state.get("last_update")
    xg_note = "xG data: placeholder"
    if last:
        st.success(f"Last updated: {last} · level: {st.session_state.get('last_level','—')} · {xg_note}")
    else:
        st.caption("Tip: click a refresh button to pull the latest FPL & fixture data.")

    if st.session_state.get("last_level") in ("full", "heavy"):
        st.info(
            "ℹ️ Full/Heavy refresh will add xG (Full) and injury scraping (Heavy) — "
            "these are placeholders in the MVP and currently behave like Quick refresh."
        )

    tab1, tab2, tab3 = st.tabs(
        ["📅 Next Gameweek", "🏆 FPL Recommendations", "🌍 Predictions (All Leagues)"]
    )
    with tab1:
        tab_next_gameweek(trained, history, cfg["api_key"])
    with tab2:
        tab_fpl(trained, cfg["api_key"], cfg["team_id"])
    with tab3:
        tab_all_leagues(models, histories, cfg["api_key"])

    st.divider()
    st.caption("⚠️ Not financial advice. Gamble responsibly. Predictions are probabilistic and for entertainment.")


@st.cache_data(ttl=300, show_spinner=False)
def _validate_key_cached(api_key: str):
    return data_fetch.validate_org_key(api_key)


if __name__ == "__main__":
    main()
