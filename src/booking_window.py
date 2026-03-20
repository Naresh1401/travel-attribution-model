"""
Booking Window Analysis
========================
Travel-specific attribution analysis segmented by booking lead time.

Segments:
- Last-minute: <7 days between first touch and booking
- Short:       7-30 days
- Long:        30-90 days
- Early:       90+ days

Key findings embedded in this analysis:
- Last-minute bookers convert via direct/brand search (short, direct paths)
- Long-lead bookers need 3× more social + metasearch touchpoints
- Budget allocation should differ by booking window segment
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


LEAD_TIME_SEGMENTS = {
    "last_minute": (0, 7),
    "short": (7, 30),
    "long": (30, 90),
    "early": (90, float("inf")),
}

SEGMENT_ORDER = ["last_minute", "short", "long", "early"]
SEGMENT_LABELS = {
    "last_minute": "Last-minute (<7 days)",
    "short": "Short (7-30 days)",
    "long": "Long (30-90 days)",
    "early": "Early bird (90+ days)",
}


def assign_lead_time_segment(lead_time: int | float) -> str:
    """Classify a booking lead time into a segment."""
    for seg, (lo, hi) in LEAD_TIME_SEGMENTS.items():
        if lo <= lead_time < hi:
            return seg
    return "early"


def segment_journeys(df: pd.DataFrame) -> pd.DataFrame:
    """Add lead_time_segment column to converting journey data."""
    converting_ids = df.loc[df["converted"] == 1, "journey_id"].unique()
    cdf = df[df["journey_id"].isin(converting_ids)].copy()

    # Get lead time per journey
    lead_times = (
        cdf.loc[cdf["converted"] == 1, ["journey_id", "booking_lead_time"]]
        .drop_duplicates("journey_id")
        .set_index("journey_id")["booking_lead_time"]
    )
    cdf["lead_time_segment"] = cdf["journey_id"].map(lead_times).apply(assign_lead_time_segment)
    return cdf


def attribution_by_segment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute linear attribution within each booking window segment.
    Returns channel × segment breakdown of attributed revenue.
    """
    cdf = segment_journeys(df)

    # Booking value per journey
    bv_map = (
        cdf.loc[cdf["converted"] == 1, ["journey_id", "booking_value"]]
        .drop_duplicates("journey_id")
        .set_index("journey_id")["booking_value"]
    )
    cdf["journey_bv"] = cdf["journey_id"].map(bv_map)
    cdf["credit"] = cdf["journey_bv"] / cdf["total_touchpoints"]

    result = (
        cdf.groupby(["lead_time_segment", "channel"])
        .agg(
            attributed_revenue=("credit", "sum"),
            touchpoints=("credit", "count"),
        )
        .reset_index()
    )

    # Add segment totals for percentage
    seg_totals = result.groupby("lead_time_segment")["attributed_revenue"].sum().rename("segment_total")
    result = result.merge(seg_totals, on="lead_time_segment")
    result["revenue_pct"] = result["attributed_revenue"] / result["segment_total"] * 100

    return result


def segment_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summary statistics per booking window segment."""
    cdf = segment_journeys(df)

    journey_stats = (
        cdf.groupby("journey_id")
        .agg(
            lead_time_segment=("lead_time_segment", "first"),
            n_touchpoints=("touchpoint_number", "max"),
            booking_value=("booking_value", "max"),
            n_channels=("channel", "nunique"),
            n_devices=("device", "nunique"),
        )
    )

    summary = (
        journey_stats.groupby("lead_time_segment")
        .agg(
            n_journeys=("booking_value", "count"),
            avg_touchpoints=("n_touchpoints", "mean"),
            avg_booking_value=("booking_value", "mean"),
            avg_channels=("n_channels", "mean"),
            avg_devices=("n_devices", "mean"),
        )
        .round(2)
    )

    # Reorder
    summary = summary.reindex(SEGMENT_ORDER).reset_index()
    summary["segment_label"] = summary["lead_time_segment"].map(SEGMENT_LABELS)

    return summary


def plot_channel_mix_by_segment(df: pd.DataFrame) -> go.Figure:
    """Stacked bar chart: channel mix by booking window segment."""
    attr = attribution_by_segment(df)

    # Pivot to channel × segment
    pivot = attr.pivot_table(
        index="lead_time_segment",
        columns="channel",
        values="revenue_pct",
        aggfunc="first",
        fill_value=0,
    ).reindex(SEGMENT_ORDER)

    fig = go.Figure()

    colors = px.colors.qualitative.Set3
    for i, ch in enumerate(pivot.columns):
        fig.add_trace(go.Bar(
            x=[SEGMENT_LABELS.get(s, s) for s in pivot.index],
            y=pivot[ch],
            name=ch,
            marker_color=colors[i % len(colors)],
        ))

    fig.update_layout(
        barmode="stack",
        title="Channel Mix by Booking Window Segment",
        xaxis_title="Booking Window",
        yaxis_title="Revenue Share (%)",
        legend_title="Channel",
        template="plotly_white",
        height=500,
    )
    return fig


def plot_touchpoint_comparison(df: pd.DataFrame) -> go.Figure:
    """Average touchpoints per segment — shows that long-lead needs more touches."""
    summary = segment_summary(df)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=summary["segment_label"],
        y=summary["avg_touchpoints"],
        marker_color=["#e74c3c", "#f39c12", "#3498db", "#2ecc71"],
        text=summary["avg_touchpoints"].round(1),
        textposition="auto",
    ))

    fig.update_layout(
        title="Average Touchpoints by Booking Window",
        xaxis_title="Booking Window",
        yaxis_title="Avg Touchpoints",
        template="plotly_white",
        height=400,
    )
    return fig


def budget_recommendation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate budget reallocation recommendations based on booking window analysis.
    Compares current spend proxy (last-click) vs. recommended (linear by segment).
    """
    attr = attribution_by_segment(df)

    # Aggregate across segments for recommended allocation
    recommended = (
        attr.groupby("channel")["attributed_revenue"]
        .sum()
        .reset_index()
        .rename(columns={"attributed_revenue": "recommended_revenue"})
    )
    total_rec = recommended["recommended_revenue"].sum()
    recommended["recommended_pct"] = recommended["recommended_revenue"] / total_rec * 100

    # Last-click proxy for "current" allocation
    converting_ids = df.loc[df["converted"] == 1, "journey_id"].unique()
    cdf = df[df["journey_id"].isin(converting_ids)]
    last_touches = cdf.sort_values("session_timestamp").groupby("journey_id").last().reset_index()
    current = (
        last_touches.groupby("channel")["booking_value"]
        .sum()
        .reset_index()
        .rename(columns={"booking_value": "current_revenue"})
    )
    total_cur = current["current_revenue"].sum()
    current["current_pct"] = current["current_revenue"] / total_cur * 100

    merged = recommended.merge(current, on="channel", how="outer").fillna(0)
    merged["delta_pct"] = merged["recommended_pct"] - merged["current_pct"]
    merged["recommendation"] = merged["delta_pct"].apply(
        lambda d: "↑ Increase" if d > 2 else ("↓ Decrease" if d < -2 else "→ Maintain")
    )

    return merged.sort_values("delta_pct", ascending=False)


if __name__ == "__main__":
    from pathlib import Path

    data_path = Path("data/travel_journeys.csv")
    if not data_path.exists():
        from src.generate_data import generate_journeys, save_data
        df = generate_journeys()
        save_data(df)
    else:
        df = pd.read_csv(data_path, parse_dates=["session_timestamp"])

    print("\n=== Booking Window Analysis ===\n")

    summary = segment_summary(df)
    print("--- Segment Summary ---")
    print(summary[["segment_label", "n_journeys", "avg_touchpoints",
                    "avg_booking_value", "avg_channels"]].to_string(index=False))

    print("\n--- Channel Mix by Segment ---")
    attr = attribution_by_segment(df)
    pivot = attr.pivot_table(
        index="lead_time_segment",
        columns="channel",
        values="revenue_pct",
        aggfunc="first",
        fill_value=0,
    ).reindex(SEGMENT_ORDER).round(1)
    print(pivot.to_string())

    print("\n--- Budget Recommendations ---")
    recs = budget_recommendation(df)
    print(recs[["channel", "current_pct", "recommended_pct", "delta_pct", "recommendation"]]
          .round(1).to_string(index=False))
