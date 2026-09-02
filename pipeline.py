"""
Data pipeline for the South Korea Top 50 Comeback Momentum & Fandom Intensity app.
This replicates, cleans up, and vectorizes the analysis from the original notebook:
  - EDA cleaning (dedupe, dtypes, text normalization)
  - Data validation (50 entries/day, popularity bounds)
  - Chart re-entry detection & chart-run segmentation
  - Momentum spike measurement (popularity change, rank jump, momentum score)
  - Momentum sustainability (decay speed, post-peak volatility)
  - Fandom Engagement Intensity Index (per track)
"""

import numpy as np
import pandas as pd


def load_and_process(csv_path_or_buffer) -> pd.DataFrame:
    """Run the full cleaning + feature engineering pipeline on the raw CSV."""

    df = pd.read_csv(csv_path_or_buffer)

    # ---- 1. Basic cleaning -------------------------------------------------
    df = df.drop_duplicates(keep="first")
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
    df = df.dropna(subset=["date"])
    df["is_explicit"] = df["is_explicit"].astype(bool)

    for col in ["song", "artist", "album_type"]:
        df[col] = df[col].astype(str).str.strip()
    df["album_type"] = df["album_type"].astype("category")

    df["position"] = pd.to_numeric(df["position"], downcast="unsigned")
    df["popularity"] = pd.to_numeric(df["popularity"], downcast="unsigned")
    df["duration_ms"] = pd.to_numeric(df["duration_ms"], downcast="unsigned")
    df["total_tracks"] = pd.to_numeric(df["total_tracks"], downcast="unsigned")

    # ---- 2. Normalize identifiers & derived columns ------------------------
    df["song_clean"] = df["song"].str.lower()
    df["artist_clean"] = df["artist"].str.lower()
    df["song_artist_id"] = df["artist_clean"] + "|||" + df["song_clean"]
    df["duration_min"] = df["duration_ms"] / 60000.0

    # Fix casing mismatches: use the most common casing per identifier
    df["song"] = df.groupby("song_artist_id")["song"].transform(
        lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]
    )
    df["artist"] = df.groupby("song_artist_id")["artist"].transform(
        lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]
    )

    # ---- 3. Validate popularity bounds --------------------------------------
    invalid_mask = (df["popularity"] < 0) | (df["popularity"] > 100)
    if invalid_mask.any():
        track_medians = df.groupby("song_artist_id")["popularity"].transform("median")
        df["popularity"] = np.where(
            invalid_mask, track_medians.fillna(df["popularity"].median()), df["popularity"]
        )
    df["popularity"] = df["popularity"].astype("uint8")
    df["position"] = df["position"].astype("uint8")

    # ---- 4. Enforce exactly 50 entries/day ----------------------------------
    df = df.sort_values(by=["date", "position"])
    df = df.groupby("date", observed=True).head(50).reset_index(drop=True)
    daily_counts = df.groupby("date", observed=True).size()
    corrupted_dates = daily_counts[daily_counts < 50].index
    if len(corrupted_dates) > 0:
        df = df[~df["date"].isin(corrupted_dates)].reset_index(drop=True)

    # ---- 5. Chart re-entry detection & run segmentation ---------------------
    df = df.sort_values(by=["song_artist_id", "date"]).reset_index(drop=True)
    df["prev_date"] = df.groupby("song_artist_id")["date"].shift(1)
    df["time_gap_days"] = (df["date"] - df["prev_date"]).dt.days
    df["is_reentry"] = df["time_gap_days"] > 1
    df["chart_run_id"] = df.groupby("song_artist_id")["is_reentry"].cumsum().astype("uint16")
    df["run_type"] = np.where(df["chart_run_id"] == 0, "Organic", "Comeback")
    df["days_in_current_run"] = (
        df.groupby(["song_artist_id", "chart_run_id"]).cumcount() + 1
    )

    # ---- 6. Momentum spike measurement --------------------------------------
    df["prev_popularity"] = df.groupby("song_artist_id")["popularity"].shift(1)
    df["prev_position"] = df.groupby("song_artist_id")["position"].shift(1)
    df["popularity_change_rate"] = df["popularity"] - df["prev_popularity"]
    df["rank_jump_magnitude"] = df["prev_position"] - df["position"]
    df["peak_rank_after_reentry"] = df.groupby(["song_artist_id", "chart_run_id"])[
        "position"
    ].transform("min")
    df["momentum_intensity_score"] = df["popularity_change_rate"] + df["rank_jump_magnitude"]
    df.loc[
        df["prev_date"].isna(),
        ["popularity_change_rate", "rank_jump_magnitude", "momentum_intensity_score"],
    ] = 0

    # ---- 7. Momentum sustainability -----------------------------------------
    df["days_retained_in_run"] = df.groupby(["song_artist_id", "chart_run_id"])[
        "date"
    ].transform("count")
    df["is_peak_row"] = df["position"] == df["peak_rank_after_reentry"]
    peak_dates = df.where(df["is_peak_row"])["date"]
    df["peak_date_in_run"] = peak_dates.groupby(
        [df["song_artist_id"], df["chart_run_id"]]
    ).transform("first")
    df["days_post_peak"] = (df["date"] - df["peak_date_in_run"]).dt.days
    df["rank_drop_from_peak"] = df["position"] - df["peak_rank_after_reentry"]
    df["rank_decay_speed"] = np.where(
        df["days_post_peak"] > 0, df["rank_drop_from_peak"] / df["days_post_peak"], 0
    )
    post_peak = df[df["days_post_peak"] >= 0]
    vol = post_peak.groupby(["song_artist_id", "chart_run_id"])["position"].std()
    df = df.merge(
        vol.rename("post_peak_volatility"),
        on=["song_artist_id", "chart_run_id"],
        how="left",
    )
    df["post_peak_volatility"] = df["post_peak_volatility"].fillna(0)

    return df


def min_max_scale(series: pd.Series) -> pd.Series:
    if series.max() == series.min():
        return pd.Series(0, index=series.index)
    return (series - series.min()) / (series.max() - series.min())


def build_fandom_index(df: pd.DataFrame) -> pd.DataFrame:
    """Per-track Fandom Engagement Intensity Index (0-100).
    Weighting: 40% spike velocity, 40% recovery speed, 20% re-entry frequency.
    """
    fandom_base = df.groupby(["artist", "song", "song_artist_id"], observed=True).agg(
        reentry_frequency=("is_reentry", "sum"),
        max_popularity_spike=("popularity_change_rate", "max"),
        fastest_rank_recovery=("rank_jump_magnitude", "max"),
        avg_position=("position", "mean"),
        avg_popularity=("popularity", "mean"),
    ).reset_index()

    fandom_base["freq_score"] = min_max_scale(fandom_base["reentry_frequency"])
    fandom_base["spike_score"] = min_max_scale(fandom_base["max_popularity_spike"])
    fandom_base["recovery_score"] = min_max_scale(fandom_base["fastest_rank_recovery"])

    fandom_base["fandom_engagement_intensity_index"] = (
        fandom_base["freq_score"] * 0.20
        + fandom_base["spike_score"] * 0.40
        + fandom_base["recovery_score"] * 0.40
    ) * 100

    return fandom_base.sort_values("fandom_engagement_intensity_index", ascending=False)
