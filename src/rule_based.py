"""
Rule-Based Attribution Models for Travel Journeys
===================================================
Five industry-standard heuristic models adapted for travel:

1. Last-click   — all credit to final touchpoint (brand search hero problem)
2. First-click  — all credit to first touchpoint (awareness bias)
3. Linear       — equal credit to every touchpoint
4. Time-decay   — 14-day half-life (suitable for long travel research windows)
5. W-shaped     — 30/10/30/30 position-based for travel's 3-phase journey
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _get_converting_journeys(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to journeys that ended in a conversion."""
    converting_ids = df.loc[df["converted"] == 1, "journey_id"].unique()
    return df[df["journey_id"].isin(converting_ids)].copy()


def last_click(df: pd.DataFrame) -> pd.DataFrame:
    """100% credit to the last touchpoint before conversion."""
    cdf = _get_converting_journeys(df)
    last_touches = cdf.sort_values("session_timestamp").groupby("journey_id").last().reset_index()
    result = (
        last_touches.groupby("channel")["booking_value"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "attributed_revenue", "count": "attributed_conversions"})
        .reset_index()
    )
    result["model"] = "last_click"
    return result


def first_click(df: pd.DataFrame) -> pd.DataFrame:
    """100% credit to the first touchpoint in the journey."""
    cdf = _get_converting_journeys(df)

    # Map booking value from converting row to all rows in journey
    bv_map = (
        cdf.loc[cdf["converted"] == 1, ["journey_id", "booking_value"]]
        .drop_duplicates("journey_id")
        .set_index("journey_id")["booking_value"]
    )

    first_touches = cdf.sort_values("session_timestamp").groupby("journey_id").first().reset_index()
    first_touches["booking_value"] = first_touches["journey_id"].map(bv_map)

    result = (
        first_touches.groupby("channel")["booking_value"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "attributed_revenue", "count": "attributed_conversions"})
        .reset_index()
    )
    result["model"] = "first_click"
    return result


def linear(df: pd.DataFrame) -> pd.DataFrame:
    """Equal credit split across all touchpoints in converting journeys."""
    cdf = _get_converting_journeys(df)

    # Assign booking_value from the converting (last) row to all rows in journey
    bv_map = (
        cdf.loc[cdf["converted"] == 1, ["journey_id", "booking_value"]]
        .drop_duplicates("journey_id")
        .set_index("journey_id")["booking_value"]
    )
    cdf = cdf.copy()
    cdf["journey_bv"] = cdf["journey_id"].map(bv_map)
    cdf["credit"] = cdf["journey_bv"] / cdf["total_touchpoints"]

    result = (
        cdf.groupby("channel")["credit"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "attributed_revenue", "count": "attributed_conversions"})
        .reset_index()
    )
    result["model"] = "linear"
    return result


def time_decay(df: pd.DataFrame, half_life_days: float = 14.0) -> pd.DataFrame:
    """
    Exponential time-decay with 14-day half-life.
    More recent touchpoints receive more credit — suitable for
    travel's 2-8 week research journeys.
    """
    cdf = _get_converting_journeys(df).copy()

    bv_map = (
        cdf.loc[cdf["converted"] == 1, ["journey_id", "booking_value"]]
        .drop_duplicates("journey_id")
        .set_index("journey_id")["booking_value"]
    )
    cdf["journey_bv"] = cdf["journey_id"].map(bv_map)

    # Conversion timestamp per journey
    conv_ts = (
        cdf.loc[cdf["converted"] == 1, ["journey_id", "session_timestamp"]]
        .drop_duplicates("journey_id")
        .rename(columns={"session_timestamp": "conversion_ts"})
        .set_index("journey_id")["conversion_ts"]
    )
    cdf["conversion_ts"] = cdf["journey_id"].map(conv_ts)
    cdf["days_before_conv"] = (
        (cdf["conversion_ts"] - cdf["session_timestamp"]).dt.total_seconds() / 86400
    )

    # Decay weight: w = 2^(-days / half_life)
    decay_rate = np.log(2) / half_life_days
    cdf["weight"] = np.exp(-decay_rate * cdf["days_before_conv"].clip(lower=0))

    # Normalise weights within each journey
    journey_weight_sum = cdf.groupby("journey_id")["weight"].transform("sum")
    cdf["credit"] = cdf["journey_bv"] * cdf["weight"] / journey_weight_sum

    result = (
        cdf.groupby("channel")["credit"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "attributed_revenue", "count": "attributed_conversions"})
        .reset_index()
    )
    result["model"] = "time_decay"
    return result


def w_shaped(df: pd.DataFrame) -> pd.DataFrame:
    """
    W-shaped (position-based) model adapted for travel's 3-phase journey:
      - 30% to first touch  (awareness / inspiration)
      - 30% to middle touch (lead-gen / metasearch consideration)
      - 30% to last touch   (conversion / booking)
      - 10% split among remaining touchpoints

    For journeys with ≤2 touchpoints, reverts to simpler splits.
    """
    cdf = _get_converting_journeys(df).copy()

    bv_map = (
        cdf.loc[cdf["converted"] == 1, ["journey_id", "booking_value"]]
        .drop_duplicates("journey_id")
        .set_index("journey_id")["booking_value"]
    )
    cdf["journey_bv"] = cdf["journey_id"].map(bv_map)

    # Sort and assign positions
    cdf = cdf.sort_values(["journey_id", "session_timestamp"])
    cdf["pos"] = cdf.groupby("journey_id").cumcount()

    credits = []
    for jid, group in cdf.groupby("journey_id"):
        n = len(group)
        bv = group["journey_bv"].iloc[0]

        if n == 1:
            w = np.array([1.0])
        elif n == 2:
            w = np.array([0.50, 0.50])
        else:
            w = np.full(n, 0.0)
            mid = n // 2
            w[0] = 0.30    # first touch
            w[mid] = 0.30  # lead-gen / mid touch
            w[-1] = 0.30   # booking touch
            remaining = n - 3 if n > 3 else 0
            if remaining > 0:
                remaining_indices = [i for i in range(n) if i not in (0, mid, n - 1)]
                for ri in remaining_indices:
                    w[ri] = 0.10 / remaining
            else:
                # n == 3: first/mid/last already set to 0.30 each → total 0.90
                # distribute remaining 0.10 equally
                w += 0.10 / 3

        credits.extend(bv * w)

    cdf["credit"] = credits

    result = (
        cdf.groupby("channel")["credit"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "attributed_revenue", "count": "attributed_conversions"})
        .reset_index()
    )
    result["model"] = "w_shaped"
    return result


def run_all_models(df: pd.DataFrame) -> pd.DataFrame:
    """Run all 5 rule-based models and return a combined DataFrame."""
    results = pd.concat([
        last_click(df),
        first_click(df),
        linear(df),
        time_decay(df),
        w_shaped(df),
    ], ignore_index=True)

    # Add percentage columns
    for model_name in results["model"].unique():
        mask = results["model"] == model_name
        total_rev = results.loc[mask, "attributed_revenue"].sum()
        results.loc[mask, "revenue_pct"] = results.loc[mask, "attributed_revenue"] / total_rev * 100

    return results


if __name__ == "__main__":
    from pathlib import Path

    data_path = Path("data/travel_journeys.csv")
    if not data_path.exists():
        from src.generate_data import generate_journeys, save_data
        df = generate_journeys()
        save_data(df)
    else:
        df = pd.read_csv(data_path, parse_dates=["session_timestamp"])

    results = run_all_models(df)

    print("\n=== Attribution Model Comparison ===\n")
    pivot = results.pivot_table(
        index="channel",
        columns="model",
        values="revenue_pct",
        aggfunc="first",
    ).round(1)
    print(pivot.to_string())
    print(f"\nTotal attributed revenue per model:")
    print(results.groupby("model")["attributed_revenue"].sum().round(0).to_string())
