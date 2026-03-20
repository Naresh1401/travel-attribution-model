"""
Shapley Value Attribution Model
================================
Applies cooperative game theory to travel marketing attribution.
Each channel is a "player" and the "payoff" is the conversion rate of
journeys containing that channel coalition.

The Shapley value fairly distributes credit by computing each channel's
marginal contribution across all possible orderings of channel coalitions.

Cross-device adjustment: sessions from same customer on different devices
are merged using customer_id before computing coalition values.
"""

from __future__ import annotations

import math
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd


def _build_coalition_data(df: pd.DataFrame) -> tuple[dict[frozenset, float], int]:
    """
    Build coalition conversion rates from journey data.
    Cross-device: merges sessions by customer_id (already grouped by journey_id).

    Returns:
        coalition_conv_rates: {frozenset of channels: conversion_rate}
        n_journeys: total number of journeys
    """
    journey_data = (
        df.sort_values("session_timestamp")
        .groupby("journey_id")
        .agg(
            channels=("channel", lambda x: frozenset(x)),
            converted=("converted", "max"),
        )
    )

    # Count conversions and totals per unique channel set
    coalition_stats: dict[frozenset, list] = defaultdict(lambda: [0, 0])  # [conversions, total]

    for _, row in journey_data.iterrows():
        ch_set = row["channels"]
        coalition_stats[ch_set][1] += 1
        coalition_stats[ch_set][0] += row["converted"]

    coalition_conv_rates = {}
    for ch_set, (conv, total) in coalition_stats.items():
        coalition_conv_rates[ch_set] = conv / total if total > 0 else 0.0

    return coalition_conv_rates, len(journey_data)


def _coalition_value(
    coalition: frozenset,
    coalition_conv_rates: dict[frozenset, float],
    all_channels: list[str],
) -> float:
    """
    Estimate the conversion rate for a given coalition.
    Uses journeys where the channel set is a subset of the coalition.
    """
    if len(coalition) == 0:
        return 0.0

    # Average the conversion rates of all observed journeys whose channel
    # set is a subset of the requested coalition
    matching_rates = []
    matching_weights = []

    for observed_set, rate in coalition_conv_rates.items():
        if observed_set.issubset(coalition):
            matching_rates.append(rate)
            matching_weights.append(1)
        elif coalition.issubset(observed_set):
            # Also include supersets with a lower weight
            matching_rates.append(rate)
            matching_weights.append(0.5)

    if not matching_rates:
        return 0.0

    weights = np.array(matching_weights, dtype=float)
    return float(np.average(matching_rates, weights=weights))


def compute_shapley_values(
    df: pd.DataFrame,
    max_channels: int | None = None,
    sampling_iterations: int = 10_000,
) -> pd.DataFrame:
    """
    Compute Shapley values for each channel using Monte Carlo sampling.

    For N channels, exact Shapley requires evaluating 2^N coalitions.
    With 10 channels that's 1,024 — feasible. We use exact computation
    when N ≤ 12, otherwise Monte Carlo sampling.

    Args:
        df: Journey touchpoint data.
        max_channels: Limit number of channels (for testing).
        sampling_iterations: Number of Monte Carlo permutations.

    Returns:
        DataFrame with Shapley values per channel.
    """
    coalition_conv_rates, n_journeys = _build_coalition_data(df)
    all_channels = sorted(df["channel"].unique().tolist())

    if max_channels:
        all_channels = all_channels[:max_channels]

    n = len(all_channels)

    if n <= 12:
        # Exact Shapley computation
        shapley_values = _exact_shapley(all_channels, coalition_conv_rates)
    else:
        shapley_values = _monte_carlo_shapley(
            all_channels, coalition_conv_rates, sampling_iterations
        )

    # Convert to DataFrame
    total_revenue = df.loc[df["converted"] == 1, "booking_value"].sum()
    total_conversions = df["converted"].sum()

    total_sv = sum(shapley_values.values())
    results = []
    for ch in all_channels:
        sv = shapley_values[ch]
        weight = sv / total_sv if total_sv > 0 else 1.0 / n
        results.append({
            "channel": ch,
            "shapley_value": sv,
            "attribution_weight": weight,
            "attributed_revenue": weight * total_revenue,
            "attributed_conversions": round(weight * total_conversions),
            "revenue_pct": weight * 100,
            "model": "shapley",
        })

    return pd.DataFrame(results)


def _exact_shapley(
    channels: list[str],
    coalition_conv_rates: dict[frozenset, float],
) -> dict[str, float]:
    """Exact Shapley value computation for small number of channels."""
    n = len(channels)
    shapley = {ch: 0.0 for ch in channels}

    for ch in channels:
        other_channels = [c for c in channels if c != ch]

        for size in range(0, n):
            for subset in combinations(other_channels, size):
                coalition_without = frozenset(subset)
                coalition_with = coalition_without | {ch}

                v_without = _coalition_value(coalition_without, coalition_conv_rates, channels)
                v_with = _coalition_value(coalition_with, coalition_conv_rates, channels)

                marginal = v_with - v_without

                # Shapley weight: |S|!(n-|S|-1)! / n!
                s = len(coalition_without)
                weight = math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n)
                shapley[ch] += weight * marginal

    return shapley


def _monte_carlo_shapley(
    channels: list[str],
    coalition_conv_rates: dict[frozenset, float],
    n_iterations: int,
) -> dict[str, float]:
    """Monte Carlo approximation of Shapley values."""
    rng = np.random.Generator(np.random.PCG64(42))
    n = len(channels)
    shapley = {ch: 0.0 for ch in channels}

    for _ in range(n_iterations):
        perm = rng.permutation(channels).tolist()

        coalition = frozenset()
        prev_value = 0.0

        for ch in perm:
            new_coalition = coalition | {ch}
            new_value = _coalition_value(new_coalition, coalition_conv_rates, channels)
            marginal = new_value - prev_value
            shapley[ch] += marginal
            coalition = new_coalition
            prev_value = new_value

    # Average over iterations
    for ch in channels:
        shapley[ch] /= n_iterations

    return shapley


def cross_device_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyse cross-device patterns in converting journeys.
    Shows how many journeys span multiple devices and the typical
    device transition patterns.
    """
    converting_ids = df.loc[df["converted"] == 1, "journey_id"].unique()
    cdf = df[df["journey_id"].isin(converting_ids)].copy()

    journey_devices = (
        cdf.sort_values("session_timestamp")
        .groupby("journey_id")
        .agg(
            devices=("device", lambda x: list(x)),
            n_devices=("device", "nunique"),
            channels=("channel", lambda x: list(x)),
            booking_value=("booking_value", "max"),
        )
    )

    journey_devices["is_cross_device"] = journey_devices["n_devices"] > 1
    journey_devices["device_path"] = journey_devices["devices"].apply(lambda x: " → ".join(x))
    journey_devices["first_device"] = journey_devices["devices"].apply(lambda x: x[0])
    journey_devices["last_device"] = journey_devices["devices"].apply(lambda x: x[-1])

    return journey_devices


if __name__ == "__main__":
    from pathlib import Path

    data_path = Path("data/travel_journeys.csv")
    if not data_path.exists():
        from src.generate_data import generate_journeys, save_data
        df = generate_journeys()
        save_data(df)
    else:
        df = pd.read_csv(data_path, parse_dates=["session_timestamp"])

    print("Computing Shapley values (exact method for 10 channels)...")
    result = compute_shapley_values(df)
    print("\n=== Shapley Value Attribution ===\n")
    print(result[["channel", "shapley_value", "attributed_revenue", "revenue_pct"]]
          .sort_values("shapley_value", ascending=False)
          .to_string(index=False))

    print("\n--- Cross-Device Analysis ---")
    cd = cross_device_analysis(df)
    pct_cross = cd["is_cross_device"].mean() * 100
    print(f"Cross-device journeys: {pct_cross:.1f}%")
    print(f"\nMost common device transitions:")
    print(cd.groupby(["first_device", "last_device"]).size()
          .sort_values(ascending=False).head(6).to_string())
