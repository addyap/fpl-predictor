# ⚽ FPL Predictor

A Streamlit app that predicts football match outcomes across **22 European leagues**
and turns Premier League predictions into Fantasy Premier League recommendations.

- **Next Gameweek** — upcoming PL fixtures with predicted winner, confidence % and odds comparison
- **FPL Recommendations** — captain shortlist (with ownership differential), 8-week fixture-difficulty radar, injury alerts; personalised to your team ID
- **All Leagues** — outcome predictions for the top leagues (live fixtures) + model-based power rankings for the rest

The model is a multinomial logistic regression (home / draw / away) trained on two
seasons of results per league from [football-data.co.uk](https://www.football-data.co.uk/).
Live fixtures come from [football-data.org](https://www.football-data.org/) and the
official FPL API.

## Run locally

```bash
pip install -r requirements.txt
streamlit run main.py
```

On first run it downloads the historical data for all leagues (a progress bar shows).
Enter your free football-data.org API key in the sidebar, or store it in
`.streamlit/secrets.toml` (see below).

## API key

Get a free key at [football-data.org](https://www.football-data.org/) (~2 min signup).

**Local:** create `.streamlit/secrets.toml`:

```toml
football_data_org_key = "YOUR_API_KEY_HERE"
# fpl_team_id = "1234567"   # optional
```

**Streamlit Community Cloud:** don't commit the key. Add it in your app's
**Settings → Secrets** using the same TOML above. `.streamlit/secrets.toml` is
gitignored so a real key never reaches GitHub.

## Personal / shareable links

The FPL team id can be passed in the URL, so each person gets their own
auto-loading link off the same deployed app:

```
https://<your-app>.streamlit.app/?team=1234567
```

Priority is URL `?team=` → secrets `fpl_team_id` → blank. Typing an id in the
sidebar also updates the URL, so you can just bookmark the page.

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → pick this repo, branch `main`, main file `main.py`.
3. Open **Advanced settings → Secrets** and paste your `football_data_org_key`.
4. Deploy. First load downloads the league data, then the app is live at `https://<app>.streamlit.app`.

## Notes & limits

- football-data.org's **free tier** serves live fixtures for ~8 leagues (top 5 + Championship, Eredivisie, Primeira Liga) and is rate-limited to 10 requests/minute. Other leagues fall back to a model-based power ranking.
- Predictions are probabilistic. **Not financial advice. Gamble responsibly.**
