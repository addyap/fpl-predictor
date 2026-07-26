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
import tracking

# --------------------------------------------------------------------------- #
# Page config
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="FPL Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="auto",  # collapsed by default on mobile
)

REFRESH_LEVELS = {
    "Quick (5s)": "quick",
    "Full (20s)": "full",
    "Heavy (2min)": "heavy",
}

# Responsive tweaks so the app reads well on both phones and desktops.
_RESPONSIVE_CSS = """
<style>
/* Wide, comfortable content on desktop; tight and readable on mobile. */
@media (max-width: 640px) {
  .block-container { padding: 1rem 0.8rem 3rem !important; }
  h1 { font-size: 1.55rem !important; }
  h2 { font-size: 1.25rem !important; }
  h3 { font-size: 1.05rem !important; }
  /* Metric numbers shrink so 3-up rows don't clip on small screens. */
  [data-testid="stMetricValue"] { font-size: 1.1rem !important; }
}
/* Let column rows wrap (stack) when they can't fit, instead of squashing. */
[data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
/* Tab labels wrap rather than overflow horizontally. */
[data-baseweb="tab-list"] { flex-wrap: wrap; }
/* Wide tables/tickers scroll inside their box; the page never scrolls sideways. */
[data-testid="stDataFrame"], [data-testid="stTable"] { overflow-x: auto; }
</style>
"""


def inject_css():
    st.markdown(_RESPONSIVE_CSS, unsafe_allow_html=True)


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


@st.cache_data(ttl=300, show_spinner=False)
def cached_entry(team_id: str):
    if not team_id or not team_id.isdigit():
        return {}
    return data_fetch.fetch_fpl_entry(int(team_id))


@st.cache_data(ttl=1800, show_spinner=False)
def cached_backtest():
    """Out-of-sample backtest of the Premier League model (cached 30 min)."""
    return model_mod.backtest_model(data_fetch.load_historical(data_fetch.PL_CODE))


def build_win_prob(trained, history, boot, fixtures) -> dict:
    """
    Map each Premier League team to our model's probability of winning its next
    fixture, keyed by FPL team name so it can feed the captain scorer. Uses the
    FPL fixture list (FPL team ids) for the next gameweek. Empty on missing data.
    """
    if not boot or not fixtures or trained is None or history is None:
        return {}
    teams = {t["id"]: t["name"] for t in boot.get("teams", [])}
    events = boot.get("events", [])
    nxt = next((e["id"] for e in events if e.get("is_next")), None)
    if nxt is None:
        nxt = next((e["id"] for e in events if e.get("is_current")), None)
    known = set(history["home_team"]) | set(history["away_team"])

    win_prob: dict[str, float] = {}
    for fx in fixtures:
        if fx.get("event") != nxt or fx.get("finished"):
            continue
        h_fpl = teams.get(fx.get("team_h"))
        a_fpl = teams.get(fx.get("team_a"))
        if not h_fpl or not a_fpl:
            continue
        home = _match_team_name(h_fpl, known)
        away = _match_team_name(a_fpl, known)
        pred = model_mod.predict_fixture(trained, history, home, away)
        win_prob[h_fpl] = pred["prob_home"] / 100.0
        win_prob[a_fpl] = pred["prob_away"] / 100.0
    return win_prob


@st.cache_data(ttl=300, show_spinner=False)
def augmented_pl_history(api_key: str):
    """
    Premier League history blended with the current season's finished results
    (football-data.org), so recent-form features reflect this season. Falls back
    to the static two-season history if the API is unavailable.
    """
    base = data_fetch.load_historical(data_fetch.PL_CODE)
    if not api_key:
        return base
    recent = cached_recent_results(api_key, data_fetch.PL_COMPETITION)
    if recent.empty:
        return base
    # Normalise football-data.org names to the historical (co.uk) naming.
    known = set(base["home_team"]) | set(base["away_team"])
    recent = recent.copy()
    recent["home_team"] = recent["home_team"].apply(lambda n: _match_team_name(n, known))
    recent["away_team"] = recent["away_team"].apply(lambda n: _match_team_name(n, known))
    recent["season"] = "current"
    combined = pd.concat([base, recent], ignore_index=True)
    combined = combined.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return combined


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
    _csv_download(df, "Download predictions (CSV)", "next_gameweek_predictions.csv")
    st.caption(
        f"Model trained on {trained.n_matches} matches · "
        f"training accuracy {trained.accuracy:.1%}. Recent-form features include "
        "this season's results when your API key is set. Odds shown for comparison only."
    )

    # Honest out-of-sample performance
    with st.expander("📏 How accurate is this model, really? (out-of-sample backtest)"):
        bt = cached_backtest()
        if bt.get("accuracy") is None:
            st.write("Not enough data to backtest.")
        else:
            b1, b2, b3 = st.columns(3)
            b1.metric("Real accuracy", f"{bt['accuracy']*100:.1f}%",
                      help="On matches the model never saw during training.")
            b2.metric("Baseline (always home)", f"{bt['baseline']*100:.1f}%")
            edge = (bt["accuracy"] - bt["baseline"]) * 100
            b3.metric("Edge over baseline", f"{edge:+.1f} pts")
            st.caption(
                f"Walk-forward test on the most recent {bt['n_test']} matches "
                "(trained only on earlier games). Calibration below — does the "
                "model's confidence match reality?"
            )
            if not bt["calibration"].empty:
                st.dataframe(
                    bt["calibration"].rename(columns={
                        "confidence_bin": "Confidence bin", "n": "Matches",
                        "predicted_avg": "Model says (%)", "actual_win_rate": "Actually won (%)",
                    }), use_container_width=True, hide_index=True)

    # ---- Live track record ---------------------------------------------------
    st.markdown("### 📈 Live track record")
    boot = cached_bootstrap()
    gw = fpl.next_gameweek_info(boot).get("next") or fpl.next_gameweek_info(boot).get("current")
    # Score any predictions whose results are now in.
    if api_key:
        tracking.update_results(cached_recent_results(api_key, data_fetch.PL_COMPETITION))

    records = _prediction_records(trained, history, upcoming)
    log_col, info_col = st.columns([1, 2])
    with log_col:
        if gw and st.button(f"📌 Log GW{gw} predictions", use_container_width=True):
            n = tracking.log_predictions(int(gw), records)
            if n:
                st.success(f"Logged {n} new prediction(s).")
            else:
                st.info("Predictions for this gameweek are already logged.")
    with info_col:
        st.caption(
            "Log a gameweek's predictions; once results are in they're scored "
            "automatically, building a real accuracy record over time."
        )

    summ = tracking.summary()
    if summ["n_logged"] == 0:
        st.write("No predictions logged yet. Log a gameweek above to start your track record.")
    else:
        t1, t2, t3 = st.columns(3)
        t1.metric("Predictions logged", summ["n_logged"])
        t2.metric("Scored so far", summ["n_scored"])
        t3.metric("Live accuracy", f"{summ['accuracy']*100:.1f}%" if summ["accuracy"] is not None else "—")
        if not summ["by_gw"].empty:
            chart = summ["by_gw"].set_index("gw")["accuracy"]
            st.line_chart(chart, height=220)
    st.caption(
        "⚠️ Track record is stored in a local file. On Streamlit Cloud's free tier "
        "the disk resets on redeploy, so long-term history needs durable storage "
        "(e.g. Supabase) — ask to wire that up when you want it permanent."
    )


def _prediction_records(trained, history, upcoming) -> list[dict]:
    """Build loggable prediction records (raw names + result letter + probs)."""
    known = set(history["home_team"]) | set(history["away_team"])
    recs = []
    for _, fx in upcoming.iterrows():
        home = _match_team_name(fx["home_team"], known)
        away = _match_team_name(fx["away_team"], known)
        pred = model_mod.predict_fixture(trained, history, home, away)
        letter = max(
            (("H", pred["prob_home"]), ("D", pred["prob_draw"]), ("A", pred["prob_away"])),
            key=lambda x: x[1],
        )[0]
        recs.append(
            {
                "home": fx["home_team"], "away": fx["away_team"], "pred": letter,
                "prob_home": pred["prob_home"], "prob_draw": pred["prob_draw"],
                "prob_away": pred["prob_away"],
            }
        )
    return recs


def _csv_download(df, label: str, filename: str):
    """Render a download button for a DataFrame as CSV (no-op if empty)."""
    if df is None or df.empty:
        return
    st.download_button(
        f"⬇️ {label}",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
    )


def tab_fpl(trained, history, api_key, team_id):
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
            entry = cached_entry(team_id)
            label = entry.get("name", f"team {team_id}")
            st.success(f"Personalised to **{label}** ({len(squad_ids)} players).")
        else:
            # Distinguish "team exists but no picks yet" from a genuinely bad id.
            entry = cached_entry(team_id)
            if entry:
                mgr = f"{entry.get('player_first_name','')} {entry.get('player_last_name','')}".strip()
                st.info(
                    f"✅ Found **{entry.get('name', team_id)}**"
                    + (f" ({mgr})" if mgr else "")
                    + ". Your squad isn't published yet — FPL only releases picks once "
                    "the season's first gameweek deadline passes, so personalised "
                    "features switch on automatically then. Showing generic picks for now."
                )
            else:
                st.warning(f"Couldn't find an FPL team with id '{team_id}'. Showing generic picks.")
    else:
        st.info("No FPL team id set — showing generic recommendations.")

    # Model win-probabilities per team (links the match model to captaincy).
    win_prob = build_win_prob(trained, history, boot, fixtures)

    # ---- Double / Blank gameweeks + chip hints -------------------------------
    st.markdown("### 📅 Double / Blank Gameweeks (next 15 GWs)")
    gw_info = fpl.next_gameweek_info(boot)
    if gw_info.get("name"):
        st.caption(f"Reference gameweek: {gw_info['name']}.")
    specials = fpl.special_gameweeks(boot, fixtures, ahead=15)
    if specials.empty:
        st.write(
            "No double, triple or blank gameweeks are scheduled in the next 15 GWs. "
            "These get confirmed by the Premier League as cup rounds cause "
            "rescheduling — check back through the season."
        )
    else:
        dgw = specials[specials["double_teams"] != "—"]
        if not dgw.empty:
            nxt = dgw.iloc[0]
            st.success(f"⭐ Next Double Gameweek: **GW{nxt['gameweek']}** — {nxt['double_teams']}")
        specials_view = fpl.personal_specials(specials, boot, squad_ids)
        rename = {
            "gameweek": "GW", "type": "Type",
            "double_teams": "Double (2 games)", "triple_teams": "Triple (3 games)",
            "blank_teams": "Blank (no game)", "your_double_players": "Your players (double)",
        }
        st.dataframe(specials_view.rename(columns=rename), use_container_width=True, hide_index=True)
        for hint in fpl.chip_hints(specials):
            st.markdown(f"- {hint}")

    # ---- Captain shortlist (model-weighted) ----------------------------------
    caps = fpl.captain_recommendations(
        boot, fixtures, weeks=8, top_n=8, squad_ids=squad_ids, win_prob=win_prob
    )
    st.markdown("### 🧢 Captain shortlist")
    st.caption("Score blends FPL expected points, form, fixture ease, an ownership "
               "differential bonus, and *our* model's win probability for the team.")
    if caps.empty:
        st.write("N/A — no eligible players found.")
    else:
        top = caps.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Top captain", f"{top['web_name']}", f"{top['team_name']}")
        c2.metric("Expected pts (next)", f"{top['ep_next']:.1f}")
        c3.metric("Model win prob", f"{top['model_win_prob']*100:.0f}%")
        st.dataframe(
            caps.rename(
                columns={
                    "web_name": "Player", "team_name": "Team", "position": "Pos",
                    "now_cost": "£m", "selected_by_percent": "Owned %",
                    "form": "Form", "ep_next": "xPts", "fixture_ease": "Fixture ease",
                    "ownership_diff": "Diff bonus", "model_win_prob": "Win prob",
                    "captain_score": "Score",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        _csv_download(caps, "Download captain shortlist (CSV)", "captain_shortlist.csv")

    # ---- Price-change watch --------------------------------------------------
    st.markdown("### 💷 Price-change watch (tonight)")
    st.caption("Ranked by net transfers this gameweek — FPL hides the exact rise/"
               "fall threshold, so treat as momentum, not a guarantee.")
    risers, fallers = fpl.price_change_watch(boot, top_n=8)
    pc1, pc2 = st.columns(2)
    with pc1:
        st.markdown("**📈 Likely risers**")
        if risers.empty or risers["net_transfers_event"].max() == 0:
            st.write("No transfer momentum yet (pre-season / early).")
        else:
            st.dataframe(
                risers.rename(columns={
                    "web_name": "Player", "team_name": "Team", "position": "Pos",
                    "now_cost": "£m", "net_transfers_event": "Net transfers",
                    "cost_change_event": "Δ£ this GW",
                }), use_container_width=True, hide_index=True)
    with pc2:
        st.markdown("**📉 Likely fallers**")
        if fallers.empty:
            st.write("No notable outflows yet.")
        else:
            st.dataframe(
                fallers.rename(columns={
                    "web_name": "Player", "team_name": "Team", "position": "Pos",
                    "now_cost": "£m", "net_transfers_event": "Net transfers",
                    "cost_change_event": "Δ£ this GW",
                }), use_container_width=True, hide_index=True)

    # ---- Transfer & differential targets -------------------------------------
    st.markdown("### 🔁 Transfer targets (best value per position)")
    tt = fpl.transfer_targets(boot, fixtures, weeks=6, top_n=5)
    if tt.empty:
        st.write("N/A — player data unavailable.")
    else:
        st.dataframe(
            tt.rename(columns={
                "web_name": "Player", "team_name": "Team", "position": "Pos",
                "now_cost": "£m", "form": "Form", "ep_next": "xPts",
                "selected_by_percent": "Owned %", "value": "xPts/£m",
                "fixture_ease": "Fixture ease",
            }), use_container_width=True, hide_index=True)
        _csv_download(tt, "Download transfer targets (CSV)", "transfer_targets.csv")

    # ---- Personalised transfer suggestions (needs team id) -------------------
    st.markdown("### ⭐ Your transfer suggestions")
    if not squad_ids:
        st.info("Set your FPL team ID in the sidebar to get transfers tailored to *your* squad.")
    else:
        sugg = fpl.transfer_suggestions(boot, fixtures, squad_ids, weeks=6, n=5)
        if sugg.empty:
            st.write("Your squad already looks strong — no clearly better swap found right now.")
        else:
            st.caption("The swaps that most improve your projected points. "
                       "Δ£ is the price difference (we can't see your bank).")
            st.dataframe(
                sugg.rename(columns={
                    "out_player": "Transfer OUT", "out_team": "From",
                    "in_player": "Transfer IN", "in_team": "To",
                    "position": "Pos", "gain": "Proj. gain", "cost_delta": "Δ£m",
                }), use_container_width=True, hide_index=True)
            _csv_download(sugg, "Download suggestions (CSV)", "transfer_suggestions.csv")

    # ---- Expected goals leaders ----------------------------------------------
    st.markdown("### 📊 Expected goals & assists (xGI leaders)")
    st.caption("Official FPL expected stats — season totals, so zero until games are played.")
    xl = fpl.xg_leaders(boot, top_n=10, squad_ids=squad_ids)
    if xl.empty:
        st.write("N/A.")
    else:
        st.dataframe(
            xl.rename(columns={
                "web_name": "Player", "team_name": "Team", "position": "Pos",
                "xg": "xG", "xa": "xA", "xgi": "xGI", "xgi_per_90": "xGI/90",
            }), use_container_width=True, hide_index=True)

    # ---- Set-piece & penalty takers ------------------------------------------
    st.markdown("### 🎯 Penalty & set-piece takers")
    st.caption("First-choice takers (a big hidden points edge). Penalty takers listed first.")
    spt = fpl.set_piece_takers(boot, squad_ids=squad_ids)
    if spt.empty:
        st.write("None listed" + (" in your squad." if squad_ids else " yet."))
    else:
        st.dataframe(
            spt.rename(columns={
                "web_name": "Player", "team_name": "Team",
                "position": "Pos", "duties": "Duties",
            }), use_container_width=True, hide_index=True)

    # ---- Fixture difficulty: radar + ticker ----------------------------------
    st.markdown("### 🎯 Fixture difficulty (next 8 GWs)")
    fdr = fpl.fixture_difficulty(boot, fixtures, weeks=8)
    if fdr.empty:
        st.write("N/A — fixture data unavailable.")
    else:
        _render_fdr_radar(fdr, squad_ids, boot)

    st.markdown("#### 🗓️ Fixture ticker (all teams × next 8 GWs)")
    st.caption("Each cell: difficulty (1 easy – 5 hard), opponent, (H)ome/(A)way. '—' = blank GW.")
    ticker = fpl.fixture_ticker(boot, fixtures, weeks=8)
    if ticker.empty:
        st.write("N/A.")
    else:
        st.dataframe(ticker, use_container_width=True, hide_index=True)
        _csv_download(ticker, "Download fixture ticker (CSV)", "fixture_ticker.csv")

    # ---- Injury alerts -------------------------------------------------------
    st.markdown("### 🚑 Injury / availability alerts")
    inj = fpl.injury_alerts(boot, squad_ids=squad_ids)
    if inj.empty:
        st.write("No availability concerns" + (" in your squad." if squad_ids else "."))
    else:
        st.dataframe(
            inj.rename(columns={
                "web_name": "Player", "team_name": "Team",
                "availability": "Status", "news": "News",
            }), use_container_width=True, hide_index=True)

    # ---- Optimal squad builder -----------------------------------------------
    st.markdown("### 🧮 Optimal squad builder")
    budget = st.slider("Budget (£m)", min_value=80.0, max_value=110.0, value=100.0, step=0.5)
    sq = fpl.optimal_squad(boot, fixtures, budget=budget, weeks=6)
    if sq["squad"].empty:
        st.write("N/A — player data unavailable.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Squad cost", f"£{sq['total_cost']}m")
        m2.metric("Projected pts (next GW)", f"{sq['total_ep']}")
        m3.metric("Valid 15?", "✅ Yes" if sq["feasible"] else "⚠️ Partial")
        st.dataframe(
            sq["squad"].rename(columns={
                "web_name": "Player", "team_name": "Team", "position": "Pos",
                "now_cost": "£m", "ep_next": "xPts", "form": "Form",
                "fixture_ease": "Fixture ease",
            }), use_container_width=True, hide_index=True)
        _csv_download(sq["squad"], "Download optimal squad (CSV)", "optimal_squad.csv")
        st.caption("2 GK · 5 DEF · 5 MID · 3 FWD · max 3 per club. Projection = FPL "
                   "expected points + recent form + fixture ease.")


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
    inject_css()
    st.title("⚽ FPL Predictor")
    st.caption("Premier League outcome model + Fantasy Premier League recommendations.")

    if not ensure_historical():
        st.stop()

    cfg = sidebar()

    with st.spinner("Loading / training models for all leagues (first run only)…"):
        models, histories = get_all_models()
    trained = models.get(data_fetch.PL_CODE)
    if trained is None or histories.get(data_fetch.PL_CODE) is None:
        st.error("Premier League model unavailable. Try reloading the page.")
        st.stop()
    # Blend this season's results into the PL history used for predictions.
    history = augmented_pl_history(cfg["api_key"])

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
    if b2.button("🔄 Full Refresh", use_container_width=True, help="Adds xG-based stats"):
        do_refresh("full", cfg["api_key"])
    if b3.button("🔄 Heavy Refresh", use_container_width=True, help="Adds injuries & price watch"):
        do_refresh("heavy", cfg["api_key"])

    last = st.session_state.get("last_update")
    if last:
        st.success(f"Last updated: {last} · level: {st.session_state.get('last_level','—')} · xG: live (FPL API)")
    else:
        st.caption("Tip: click a refresh button to pull the latest FPL & fixture data.")

    tab1, tab2, tab3 = st.tabs(
        ["📅 Next Gameweek", "🏆 FPL Recommendations", "🌍 Predictions (All Leagues)"]
    )
    with tab1:
        tab_next_gameweek(trained, history, cfg["api_key"])
    with tab2:
        tab_fpl(trained, history, cfg["api_key"], cfg["team_id"])
    with tab3:
        tab_all_leagues(models, histories, cfg["api_key"])

    st.divider()
    st.caption("⚠️ Not financial advice. Gamble responsibly. Predictions are probabilistic and for entertainment.")


@st.cache_data(ttl=300, show_spinner=False)
def _validate_key_cached(api_key: str):
    return data_fetch.validate_org_key(api_key)


if __name__ == "__main__":
    main()
