"""
Markov Chain Attribution Model
===============================
Models the travel customer journey as a Markov chain where each state is a
marketing channel. Attribution is computed via **removal effect**: how much
does the overall conversion probability drop when a channel is removed?

Key insight for travel: TV brand and social_inspiration have high removal
effect (they initiate journeys that wouldn't otherwise exist) despite
receiving minimal last-click credit.

Outputs:
- Transition probability matrix
- Removal effect per channel
- Directed channel flow graph (networkx)
- Attributed revenue per channel
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _build_paths(df: pd.DataFrame) -> tuple[list[list[str]], list[list[str]]]:
    """
    Convert raw touchpoint data into channel paths.
    Returns (converting_paths, non_converting_paths).
    """
    converting_ids = set(df.loc[df["converted"] == 1, "journey_id"].unique())

    converting_paths = []
    non_converting_paths = []

    for jid, group in df.sort_values("session_timestamp").groupby("journey_id"):
        path = group["channel"].tolist()
        if jid in converting_ids:
            converting_paths.append(path)
        else:
            non_converting_paths.append(path)

    return converting_paths, non_converting_paths


def build_transition_matrix(
    converting_paths: list[list[str]],
    non_converting_paths: list[list[str]],
    channels: list[str],
) -> pd.DataFrame:
    """
    Build a first-order Markov transition matrix.

    States: [Start] + channels + [Conversion, Null]
    """
    states = ["Start"] + channels + ["Conversion", "Null"]
    n_states = len(states)
    state_idx = {s: i for i, s in enumerate(states)}

    counts = np.zeros((n_states, n_states))

    for path in converting_paths:
        # Start → first channel
        counts[state_idx["Start"], state_idx[path[0]]] += 1
        for i in range(len(path) - 1):
            counts[state_idx[path[i]], state_idx[path[i + 1]]] += 1
        # Last channel → Conversion
        counts[state_idx[path[-1]], state_idx["Conversion"]] += 1

    for path in non_converting_paths:
        counts[state_idx["Start"], state_idx[path[0]]] += 1
        for i in range(len(path) - 1):
            counts[state_idx[path[i]], state_idx[path[i + 1]]] += 1
        counts[state_idx[path[-1]], state_idx["Null"]] += 1

    # Normalise rows to get probabilities
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    probs = counts / row_sums

    # Absorbing states: Conversion and Null transition to themselves
    probs[state_idx["Conversion"], :] = 0
    probs[state_idx["Conversion"], state_idx["Conversion"]] = 1
    probs[state_idx["Null"], :] = 0
    probs[state_idx["Null"], state_idx["Null"]] = 1

    return pd.DataFrame(probs, index=states, columns=states)


def _absorption_probability(transition_matrix: pd.DataFrame) -> float:
    """
    Compute probability of reaching 'Conversion' from 'Start'
    using the fundamental matrix approach for absorbing Markov chains.
    """
    states = list(transition_matrix.index)
    absorbing = {"Conversion", "Null"}
    transient = [s for s in states if s not in absorbing]

    t_idx = [states.index(s) for s in transient]
    a_idx = [states.index(s) for s in states if s in absorbing]

    P = transition_matrix.values

    # Q = sub-matrix of transient→transient transitions
    Q = P[np.ix_(t_idx, t_idx)]
    # R = sub-matrix of transient→absorbing transitions
    R = P[np.ix_(t_idx, a_idx)]

    # Fundamental matrix: N = (I - Q)^(-1)
    I = np.eye(len(transient))
    try:
        N = np.linalg.inv(I - Q)
    except np.linalg.LinAlgError:
        # Fallback: use pseudo-inverse
        N = np.linalg.pinv(I - Q)

    # Absorption probabilities: B = N * R
    B = N @ R

    # Find probability of Start reaching Conversion
    start_idx = transient.index("Start")
    absorbing_list = [s for s in states if s in absorbing]
    conv_idx = absorbing_list.index("Conversion")

    return float(B[start_idx, conv_idx])


def removal_effect(
    df: pd.DataFrame,
    channels: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute the removal effect for each channel:
    RE(channel) = 1 - P(conversion | channel removed) / P(conversion | all channels)

    Uses the standard approach: modify the transition matrix so that the
    removed channel redirects all its outgoing traffic to Null (i.e., the
    channel becomes an absorbing dead-end).

    Returns DataFrame with channels and their removal effects.
    """
    if channels is None:
        channels = sorted(df["channel"].unique().tolist())

    converting_paths, non_converting_paths = _build_paths(df)

    # Baseline conversion probability
    tm_base = build_transition_matrix(converting_paths, non_converting_paths, channels)
    p_base = _absorption_probability(tm_base)

    states = list(tm_base.index)
    null_idx = states.index("Null")

    results = []
    for ch in channels:
        # Copy the baseline transition matrix
        tm_removed = tm_base.copy()

        # Remove channel: redirect all its outgoing transitions to Null
        ch_idx = states.index(ch)
        tm_removed.iloc[ch_idx, :] = 0.0
        tm_removed.iloc[ch_idx, null_idx] = 1.0

        # Also redirect all incoming transitions to this channel → Null
        for src_idx in range(len(states)):
            if src_idx == ch_idx:
                continue
            prob_to_ch = tm_removed.iloc[src_idx, ch_idx]
            if prob_to_ch > 0:
                tm_removed.iloc[src_idx, null_idx] += prob_to_ch
                tm_removed.iloc[src_idx, ch_idx] = 0.0

        p_removed = _absorption_probability(tm_removed)

        re = 1.0 - (p_removed / p_base) if p_base > 0 else 0.0
        re = max(0.0, re)
        results.append({"channel": ch, "removal_effect": re})

    return pd.DataFrame(results)


def markov_attribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attribute revenue based on Markov removal effects.
    Each channel receives a share of total revenue proportional to its
    normalised removal effect.
    """
    re_df = removal_effect(df)

    total_removal = re_df["removal_effect"].sum()
    re_df["attribution_weight"] = re_df["removal_effect"] / total_removal

    # Total converting revenue
    total_revenue = df.loc[df["converted"] == 1, "booking_value"].sum()
    total_conversions = df["converted"].sum()

    re_df["attributed_revenue"] = re_df["attribution_weight"] * total_revenue
    re_df["attributed_conversions"] = (re_df["attribution_weight"] * total_conversions).round(0)
    re_df["revenue_pct"] = re_df["attributed_revenue"] / total_revenue * 100
    re_df["model"] = "markov"

    return re_df


def get_transition_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Public helper to return the full transition matrix."""
    channels = sorted(df["channel"].unique().tolist())
    converting_paths, non_converting_paths = _build_paths(df)
    return build_transition_matrix(converting_paths, non_converting_paths, channels)


def build_channel_flow_graph(transition_matrix: pd.DataFrame, threshold: float = 0.02) -> nx.DiGraph:
    """
    Build a NetworkX directed graph from the transition matrix.
    Only include edges with probability >= threshold.
    """
    G = nx.DiGraph()

    states = list(transition_matrix.index)
    for s in states:
        G.add_node(s)

    for i, src in enumerate(states):
        for j, dst in enumerate(states):
            prob = transition_matrix.iloc[i, j]
            if prob >= threshold and src != dst:
                G.add_edge(src, dst, weight=prob)

    return G


def plot_channel_flow(transition_matrix: pd.DataFrame, threshold: float = 0.02) -> go.Figure:
    """Create a Plotly visualisation of the channel flow graph."""
    G = build_channel_flow_graph(transition_matrix, threshold)

    pos = nx.spring_layout(G, seed=42, k=2.0)

    # Edge traces
    edge_traces = []
    annotations = []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        weight = data["weight"]

        edge_traces.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(width=max(0.5, weight * 8), color="rgba(100,100,100,0.4)"),
                hoverinfo="text",
                text=f"{u} → {v}: {weight:.1%}",
                showlegend=False,
            )
        )

    # Node trace
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    node_text = list(G.nodes())

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        marker=dict(
            size=25,
            color=["#e74c3c" if n == "Conversion" else "#2ecc71" if n == "Start"
                   else "#95a5a6" if n == "Null" else "#3498db" for n in G.nodes()],
            line=dict(width=2, color="white"),
        ),
        text=node_text,
        textposition="top center",
        textfont=dict(size=10),
        hoverinfo="text",
        showlegend=False,
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title="Travel Channel Flow — Markov Transition Graph",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        template="plotly_white",
        height=600,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


if __name__ == "__main__":
    from pathlib import Path

    data_path = Path("data/travel_journeys.csv")
    if not data_path.exists():
        from src.generate_data import generate_journeys, save_data
        df = generate_journeys()
        save_data(df)
    else:
        df = pd.read_csv(data_path, parse_dates=["session_timestamp"])

    print("Computing Markov attribution (this may take a minute)...")
    result = markov_attribution(df)
    print("\n=== Markov Chain Attribution (Removal Effect) ===\n")
    print(result[["channel", "removal_effect", "attributed_revenue", "revenue_pct"]]
          .sort_values("removal_effect", ascending=False)
          .to_string(index=False))

    print(f"\n--- Key Insight ---")
    print("Channels with high removal effect but low last-click share")
    print("are undervalued by basic attribution. In travel, TV brand")
    print("and social_inspiration typically fall into this category.")
