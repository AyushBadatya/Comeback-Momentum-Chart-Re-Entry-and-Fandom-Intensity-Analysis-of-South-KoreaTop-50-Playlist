import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pipeline import load_and_process, build_fandom_index

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="K-Pop Chart Comeback & Fandom Intensity Dashboard",
    page_icon="🎵",
    layout="wide",
)

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "data.csv")


# ============================================================================
# DATA LOADING (cached so re-running filters doesn't re-crunch 27k rows)
# ============================================================================
@st.cache_data(show_spinner="Crunching chart history...")
def get_data(file_bytes_or_path):
    return load_and_process(file_bytes_or_path)


@st.cache_data(show_spinner=False)
def get_fandom_index(df):
    return build_fandom_index(df)


st.title("🎵 Comeback Momentum, Chart Re-Entry & Fandom Intensity")
st.caption("South Korea Spotify Top 50 — interactive analysis dashboard")

# ---- Data source: bundled file, or let the user upload their own ----------
data_source = None
if os.path.exists(DEFAULT_DATA_PATH):
    data_source = DEFAULT_DATA_PATH
else:
    st.info("No bundled dataset found — upload your Atlantic_South_Korea CSV to get started.")
    uploaded = st.file_uploader("Upload chart CSV", type=["csv"])
    if uploaded is not None:
        data_source = uploaded

if data_source is None:
    st.stop()

with st.spinner("Loading and processing chart data..."):
    df = get_data(data_source)
    fandom_df = get_fandom_index(df)

# ============================================================================
# SIDEBAR — USER CAPABILITIES
# ============================================================================
st.sidebar.header("🎛️ Filters")

min_date, max_date = df["date"].min().date(), df["date"].max().date()
date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

all_artists = sorted(df["artist"].unique())
artist_filter = st.sidebar.multiselect("Artist", options=all_artists, default=[])

song_options = sorted(
    df[df["artist"].isin(artist_filter)]["song"].unique()
    if artist_filter
    else df["song"].unique()
)
song_filter = st.sidebar.multiselect("Song", options=song_options, default=[])

max_reentry = int(df.groupby("song_artist_id")["is_reentry"].sum().max())
min_reentries = st.sidebar.slider(
    "Minimum re-entry count (per track)", 0, max_reentry, 0
)

album_types = list(df["album_type"].cat.categories)
album_toggle = st.sidebar.multiselect(
    "Album type", options=album_types, default=album_types
)

st.sidebar.markdown("---")
st.sidebar.caption(f"Dataset: {df['date'].min().date()} → {df['date'].max().date()} · "
                    f"{df['song_artist_id'].nunique():,} unique tracks · {len(df):,} chart-days")

# ---- Apply filters ----------------------------------------------------------
mask = (
    (df["date"].dt.date >= start_date)
    & (df["date"].dt.date <= end_date)
    & (df["album_type"].isin(album_toggle))
)
if artist_filter:
    mask &= df["artist"].isin(artist_filter)
if song_filter:
    mask &= df["song"].isin(song_filter)

reentry_totals = df.groupby("song_artist_id")["is_reentry"].sum()
qualifying_ids = reentry_totals[reentry_totals >= min_reentries].index
mask &= df["song_artist_id"].isin(qualifying_ids)

fdf = df[mask].copy()

if fdf.empty:
    st.warning("No data matches the current filters — try widening your selection.")
    st.stop()

# ============================================================================
# TOP-LINE KPIs
# ============================================================================
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Chart-days in view", f"{len(fdf):,}")
k2.metric("Unique tracks", f"{fdf['song_artist_id'].nunique():,}")
k3.metric("Comeback rows", f"{(fdf['run_type'] == 'Comeback').sum():,}")
avg_spike = fdf.loc[fdf["is_reentry"], "days_retained_in_run"]
k4.metric("Avg comeback retention (days)",
          f"{fdf[fdf['run_type']=='Comeback']['days_retained_in_run'].mean():.1f}"
          if (fdf['run_type']=='Comeback').any() else "—")
k5.metric("Avg momentum score", f"{fdf['momentum_intensity_score'].mean():.2f}")

st.markdown("---")

# ============================================================================
# TABS — CORE MODULES
# ============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Re-entry Timeline",
    "⚡ Momentum Spikes",
    "🔁 Comeback vs Organic",
    "🎧 Content vs Momentum",
    "🏆 Fandom Leaderboard",
])

# ---- TAB 1: Re-entry Timeline Visualizer -----------------------------------
with tab1:
    st.subheader("Chart position over time, with re-entries highlighted")
    track_options = (
        fdf.sort_values("song")[["artist", "song", "song_artist_id"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    track_options["label"] = track_options["artist"] + " — " + track_options["song"]

    ranked_visible = fandom_df[fandom_df["song_artist_id"].isin(track_options["song_artist_id"])]
    default_track = (
        ranked_visible.iloc[0]["song_artist_id"]
        if not ranked_visible.empty
        else track_options.iloc[0]["song_artist_id"]
    )
    match = track_options.index[track_options["song_artist_id"] == default_track]
    default_index = int(match[0]) if len(match) else 0

    chosen_label = st.selectbox(
        "Choose a track",
        options=track_options["label"],
        index=default_index,
    )
    chosen_id = track_options.loc[track_options["label"] == chosen_label, "song_artist_id"].iloc[0]

    track_df = fdf[fdf["song_artist_id"] == chosen_id].sort_values("date")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=track_df["date"], y=track_df["position"],
        mode="lines+markers", name="Chart position",
        line=dict(color="#4C6FFF"), marker=dict(size=6),
    ))
    reentries = track_df[track_df["is_reentry"]]
    if not reentries.empty:
        fig.add_trace(go.Scatter(
            x=reentries["date"], y=reentries["position"],
            mode="markers", name="Re-entry event",
            marker=dict(color="crimson", size=13, symbol="star"),
        ))
    fig.update_yaxes(autorange="reversed", title="Chart position (1 = top)")
    fig.update_xaxes(title="Date")
    fig.update_layout(
        title=f"{chosen_label} — {len(reentries)} re-entries in selected range",
        hovermode="x unified", height=480,
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Raw run segments for this track"):
        st.dataframe(
            track_df[["date", "position", "popularity", "run_type", "chart_run_id",
                      "days_in_current_run", "peak_rank_after_reentry"]],
            use_container_width=True, hide_index=True,
        )

# ---- TAB 2: Momentum Spike Detection ---------------------------------------
with tab2:
    st.subheader("Momentum Intensity Score over time")
    st.caption("momentum_intensity_score = popularity change + rank jump magnitude "
               "(day-over-day). Spikes flag sudden fandom-driven pushes.")

    agg_choice = st.radio("View", ["Daily average (all filtered tracks)", "Top spikes (individual events)"],
                           horizontal=True)

    if agg_choice.startswith("Daily"):
        daily = fdf.groupby("date")["momentum_intensity_score"].mean().reset_index()
        fig2 = px.line(daily, x="date", y="momentum_intensity_score",
                        title="Average daily momentum intensity")
        fig2.add_hline(y=0, line_dash="dot", line_color="gray")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        top_spikes = fdf.sort_values("momentum_intensity_score", ascending=False).head(25)
        fig2 = px.bar(
            top_spikes, x="momentum_intensity_score", y=top_spikes["artist"] + " — " + top_spikes["song"],
            orientation="h", color="momentum_intensity_score", color_continuous_scale="Sunset",
            hover_data=["date", "position", "popularity"],
            title="Top 25 momentum spikes in selected range",
        )
        fig2.update_layout(yaxis_title="", xaxis_title="Momentum intensity score", height=650)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Rank decay after peak** — how fast comeback songs fall from their peak position")
    decay = fdf[(fdf["run_type"] == "Comeback") & (fdf["days_post_peak"] >= 0)]
    if not decay.empty:
        decay_curve = decay.groupby("days_post_peak")["rank_drop_from_peak"].mean().reset_index()
        fig3 = px.line(decay_curve, x="days_post_peak", y="rank_drop_from_peak",
                        markers=True, title="Average rank drop vs. days since peak")
        fig3.update_xaxes(title="Days since peak")
        fig3.update_yaxes(title="Avg positions dropped from peak")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No comeback runs with a measurable peak in the current filter selection.")

# ---- TAB 3: Comeback vs Organic (first-entry) Comparison -------------------
with tab3:
    st.subheader("Comeback runs vs. Organic (first-entry) runs")
    comp = fdf.groupby("run_type", observed=True).agg(
        avg_position=("position", "mean"),
        avg_popularity=("popularity", "mean"),
        total_days=("song", "count"),
    ).reset_index()

    c1, c2 = st.columns(2)
    with c1:
        fig4 = px.bar(comp, x="run_type", y="avg_position", color="run_type",
                       title="Average chart position (lower = better)",
                       color_discrete_map={"Comeback": "#FF6F61", "Organic": "#4C6FFF"})
        st.plotly_chart(fig4, use_container_width=True)
    with c2:
        fig5 = px.bar(comp, x="run_type", y="avg_popularity", color="run_type",
                       title="Average Spotify popularity",
                       color_discrete_map={"Comeback": "#FF6F61", "Organic": "#4C6FFF"})
        st.plotly_chart(fig5, use_container_width=True)

    fig6 = px.box(fdf, x="run_type", y="position", color="run_type",
                   title="Position distribution: Comeback vs Organic",
                   color_discrete_map={"Comeback": "#FF6F61", "Organic": "#4C6FFF"})
    fig6.update_yaxes(autorange="reversed")
    st.plotly_chart(fig6, use_container_width=True)

    st.dataframe(comp.rename(columns={
        "run_type": "Run type", "avg_position": "Avg position",
        "avg_popularity": "Avg popularity", "total_days": "Total tracked days",
    }), use_container_width=True, hide_index=True)

# ---- TAB 4: Content Attribute vs Momentum Analysis -------------------------
with tab4:
    st.subheader("Does content type predict momentum?")
    comeback_starts = fdf[(fdf["run_type"] == "Comeback") & (fdf["days_in_current_run"] == 1)]

    if comeback_starts.empty:
        st.info("No comeback start events in the current filter selection — widen filters to see this analysis.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            by_album = comeback_starts.groupby("album_type", observed=True)[
                ["momentum_intensity_score", "rank_jump_magnitude"]
            ].mean().reset_index()
            fig7 = px.bar(by_album, x="album_type", y="momentum_intensity_score",
                           title="Comeback momentum: Single vs Album tracks",
                           color="album_type")
            st.plotly_chart(fig7, use_container_width=True)
        with c2:
            by_explicit = comeback_starts.groupby("is_explicit", observed=True)[
                ["momentum_intensity_score", "days_retained_in_run"]
            ].mean().reset_index()
            by_explicit["is_explicit"] = by_explicit["is_explicit"].map({True: "Explicit", False: "Clean"})
            fig8 = px.bar(by_explicit, x="is_explicit", y="momentum_intensity_score",
                           title="Comeback momentum: Explicit vs Clean tracks",
                           color="is_explicit")
            st.plotly_chart(fig8, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            fig9 = px.scatter(
                comeback_starts, x="duration_min", y="rank_jump_magnitude",
                trendline="ols", opacity=0.6,
                title="Song duration vs. rank jump magnitude",
                labels={"duration_min": "Duration (min)", "rank_jump_magnitude": "Rank jump"},
            )
            st.plotly_chart(fig9, use_container_width=True)
        with c4:
            fig10 = px.scatter(
                comeback_starts, x="total_tracks", y="momentum_intensity_score",
                trendline="ols", opacity=0.6,
                title="Album size vs. resurgence intensity",
                labels={"total_tracks": "Total tracks on album", "momentum_intensity_score": "Momentum score"},
            )
            st.plotly_chart(fig10, use_container_width=True)

# ---- TAB 5: Fandom Intensity Leaderboard -----------------------------------
with tab5:
    st.subheader("🏆 Fandom Engagement Intensity Index")
    st.caption("Weighted index (0–100): 40% popularity spike velocity, "
               "40% fastest rank recovery, 20% historical re-entry frequency.")

    visible_ids = fdf["song_artist_id"].unique()
    fboard = fandom_df[fandom_df["song_artist_id"].isin(visible_ids)].copy()

    top_n = st.slider("Show top N tracks", 5, 50, 15)
    board = fboard.head(top_n)
    board_sorted = board.sort_values("fandom_engagement_intensity_index")

    fig11 = px.bar(
        board_sorted,
        x="fandom_engagement_intensity_index",
        y=board_sorted["artist"] + " — " + board_sorted["song"],
        orientation="h", color="fandom_engagement_intensity_index",
        color_continuous_scale="Viridis",
        title=f"Top {top_n} most fandom-driven tracks",
    )
    fig11.update_layout(yaxis_title="", xaxis_title="Fandom intensity index", height=max(400, top_n * 28))
    st.plotly_chart(fig11, use_container_width=True)

    st.dataframe(
        board[["artist", "song", "reentry_frequency", "avg_position", "avg_popularity",
               "fandom_engagement_intensity_index"]].rename(columns={
            "artist": "Artist", "song": "Song", "reentry_frequency": "Re-entries",
            "avg_position": "Avg position", "avg_popularity": "Avg popularity",
            "fandom_engagement_intensity_index": "Fandom index",
        }).round(2),
        use_container_width=True, hide_index=True,
    )

st.markdown("---")
st.caption("Built with Streamlit · Data: Spotify South Korea Top 50 daily chart")
