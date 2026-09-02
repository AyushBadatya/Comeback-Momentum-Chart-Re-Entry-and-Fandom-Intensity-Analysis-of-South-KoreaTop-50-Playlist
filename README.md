# K-Pop Chart Comeback & Fandom Intensity Dashboard

An interactive Streamlit dashboard for the "Comeback Momentum, Chart Re-Entry, and
Fandom Intensity Analysis of South Korea Top 50 Playlist" project.

## What's inside
- `app.py` — the Streamlit app (UI, filters, 5 analysis tabs)
- `pipeline.py` — data cleaning + feature engineering (re-entry detection,
  momentum scoring, fandom index) ported from the original notebook
- `data.csv` — the chart dataset (bundled so the app works out of the box;
  if you don't include it, the app will let users upload their own CSV)
- `requirements.txt` — Python dependencies

## Run it locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
This opens the app at `http://localhost:8501` in your browser.

## Get a public https:// link (free, ~2 minutes) — Streamlit Community Cloud

This is the fastest way to get a link anyone can open in Chrome, no server
management required.

1. **Create a free GitHub account** (skip if you have one) at github.com.
2. **Create a new repository** (e.g. `kpop-chart-dashboard`) and upload these
   four files: `app.py`, `pipeline.py`, `requirements.txt`, and `data.csv`.
   - Easiest way: on the repo page, click "Add file" → "Upload files", then
     drag all four files in and click "Commit changes".
3. Go to **share.streamlit.io** and sign in with your GitHub account.
4. Click **"Create app"**, pick your repository, branch `main`, and set the
   main file path to `app.py`.
5. Click **Deploy**. In under a minute you'll get a live link like:
   `https://kpop-chart-dashboard-yourname.streamlit.app`
6. Share that link with anyone — it opens directly in Chrome (or any
   browser), no login needed for viewers.

Whenever you push changes to the GitHub repo, the live app auto-updates.

### Alternative free hosts
- **Hugging Face Spaces** (choose the "Streamlit" SDK when creating a Space)
- **Render.com** (free web service, use `streamlit run app.py --server.port $PORT --server.address 0.0.0.0` as the start command)

## App features
**Core modules (tabs):**
- Re-entry timeline visualizer — per-track chart position over time with re-entries flagged
- Momentum spike detection charts — daily average & top individual spikes, plus post-peak decay curve
- Comeback vs first-entry (Organic) comparison — avg position/popularity, distribution boxplot
- Content attribute vs momentum analysis — album type, explicit content, duration, album size
- Fandom intensity leaderboard — ranked index (0–100) with adjustable Top-N

**User capabilities (sidebar):**
- Date range selector
- Song & artist filter
- Minimum re-entry count filter
- Album type toggle
