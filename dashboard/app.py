"""
Travel Attribution Dashboard
==============================
Streamlit application with 5 tabs for interactive exploration of
multi-touch attribution results.

Run: streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.generate_data import generate_journeys, save_data
from src.rule_based import run_all_models, last_click, first_click, linear, time_decay, w_shaped
from src.markov import markov_attribution, get_transition_matrix, plot_channel_flow
from src.shapley import compute_shapley_values, cross_device_analysis
from src.booking_window import (
    attribution_by_segment,
    segment_summary,
    plot_channel_mix_by_segment,
    plot_touchpoint_comparison,
    budget_recommendation,
    SEGMENT_LABELS,
    SEGMENT_ORDER,
)

# =============================================================================
# Page config
# =============================================================================
st.set_page_config(
    page_title="Travel Multi-Touch Attribution",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Data loading
# =============================================================================
@st.cache_data(show_spinner="Generating travel journey data...")
def load_data():
    data_path = ROOT / "data" / "travel_journeys.csv"
    if data_path.exists():
        df = pd.read_csv(data_path, parse_dates=["session_timestamp"])
    else:
        df = generate_journeys()
        save_data(df, str(ROOT / "data"))
    return df


@st.cache_data(show_spinner="Running rule-based models...")
def compute_rule_based(df):
    return run_all_models(df)


@st.cache_data(show_spinner="Computing Markov attribution...")
def compute_markov(df):
    return markov_attribution(df)


@st.cache_data(show_spinner="Computing Shapley values...")
def compute_shapley(df):
    return compute_shapley_values(df)


df = load_data()

# =============================================================================
# Sidebar
# =============================================================================
st.sidebar.title("✈️ Travel Attribution")
st.sidebar.markdown("---")

n_journeys = df["journey_id"].nunique()
n_conversions = df.loc[df["converted"] == 1, "journey_id"].nunique()
total_revenue = df.loc[df["converted"] == 1, "booking_value"].sum()
conv_rate = n_conversions / n_journeys * 100

st.sidebar.metric("Total Journeys", f"{n_journeys:,}")
st.sidebar.metric("Conversions", f"{n_conversions:,}")
st.sidebar.metric("Conversion Rate", f"{conv_rate:.1f}%")
st.sidebar.metric("Total Revenue", f"£{total_revenue:,.0f}")
st.sidebar.metric("Avg Booking Value", f"£{total_revenue / max(n_conversions, 1):,.0f}")

st.sidebar.markdown("---")
st.sidebar.info(
    "This dashboard analyses multi-touch attribution for travel customer "
    "journeys. The average traveller researches for 2-8 weeks across "
    "multiple devices before booking."
)

# =============================================================================
# Tabs
# =============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ Journey Explorer",
    "📊 Attribution Comparison",
    "📅 Booking Window",
    "💰 Budget Simulator",
    "📱 Device Journey",
])

# =============================================================================
# Tab 1 — Journey Explorer (Sankey)
# =============================================================================
with tab1:
    st.header("Customer Journey Explorer")
    st.markdown(
        "Sankey diagram of the **top 15 most common travel journeys** — "
        "showing how customers flow between channels from awareness to booking."
    )

    # Build Sankey from converting journeys
    converting_ids = df.loc[df["converted"] == 1, "journey_id"].unique()
    conv_df = df[df["journey_id"].isin(converting_ids)].sort_values(
        ["journey_id", "session_timestamp"]
    )

    # Build journey paths
    journey_paths = (
        conv_df.groupby("journey_id")["channel"]
        .apply(lambda x: " → ".join(x))
        .value_counts()
        .head(15)
        .reset_index()
    )
    journey_paths.columns = ["path", "count"]

    # Build Sankey data from step transitions
    all_channels = sorted(df["channel"].unique())
    max_steps = conv_df.groupby("journey_id").size().max()
    max_steps = min(max_steps, 8)  # Cap for readability

    # Create nodes: channel_step combinations
    nodes = []
    node_map = {}

    def get_node_idx(channel, step):
        key = f"{channel}_step{step}"
        if key not in node_map:
            node_map[key] = len(nodes)
            nodes.append(f"{channel} (Step {step + 1})")
        return node_map[key]

    sources, targets, values = [], [], []

    for jid, group in conv_df.groupby("journey_id"):
        channels = group["channel"].tolist()
        for i in range(min(len(channels) - 1, max_steps - 1)):
            src = get_node_idx(channels[i], i)
            tgt = get_node_idx(channels[i + 1], i + 1)
            sources.append(src)
            targets.append(tgt)
            values.append(1)

    # Aggregate duplicate edges
    edge_agg = {}
    for s, t, v in zip(sources, targets, values):
        key = (s, t)
        edge_agg[key] = edge_agg.get(key, 0) + v

    sources = [k[0] for k in edge_agg.keys()]
    targets = [k[1] for k in edge_agg.keys()]
    values = list(edge_agg.values())

    # Color mapping
    channel_colors = {
        "paid_search_brand": "#e74c3c",
        "paid_search_generic": "#c0392b",
        "metasearch": "#3498db",
        "display_retargeting": "#f39c12",
        "email_remarketing": "#9b59b6",
        "social_inspiration": "#e91e63",
        "travel_blog_affiliate": "#2ecc71",
        "direct": "#34495e",
        "organic_search": "#1abc9c",
        "tv_brand": "#ff9800",
    }

    node_colors = []
    for label in nodes:
        ch = label.split(" (Step")[0]
        node_colors.append(channel_colors.get(ch, "#95a5a6"))

    fig_sankey = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes,
            color=node_colors,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color="rgba(200,200,200,0.3)",
        ),
    )])
    fig_sankey.update_layout(
        title_text="Converting Journey Flows — Channel Transitions",
        font_size=11,
        height=650,
    )
    st.plotly_chart(fig_sankey, use_container_width=True)

    # Show top journey paths table
    st.subheader("Top 15 Converting Journey Paths")
    journey_paths["pct"] = (journey_paths["count"] / journey_paths["count"].sum() * 100).round(1)
    st.dataframe(
        journey_paths.rename(columns={"path": "Journey Path", "count": "Count", "pct": "% of Conversions"}),
        use_container_width=True,
        hide_index=True,
    )


# =============================================================================
# Tab 2 — Attribution Comparison
# =============================================================================
with tab2:
    st.header("Attribution Model Comparison")
    st.markdown(
        "Side-by-side comparison of **7 attribution models** — revealing how "
        "different models value each channel differently."
    )

    # Run all models
    rule_results = compute_rule_based(df)
    markov_results = compute_markov(df)
    shapley_results = compute_shapley(df)

    # Combine all results
    all_results = pd.concat([
        rule_results[["channel", "attributed_revenue", "revenue_pct", "model"]],
        markov_results[["channel", "attributed_revenue", "revenue_pct", "model"]],
        shapley_results[["channel", "attributed_revenue", "revenue_pct", "model"]],
    ], ignore_index=True)

    # Pivot table
    pivot = all_results.pivot_table(
        index="channel",
        columns="model",
        values="revenue_pct",
        aggfunc="first",
    ).round(1)

    model_order = ["last_click", "first_click", "linear", "time_decay", "w_shaped", "markov", "shapley"]
    pivot = pivot.reindex(columns=[m for m in model_order if m in pivot.columns])

    st.subheader("Revenue Attribution by Channel (%)")
    st.dataframe(pivot.style.background_gradient(cmap="YlOrRd", axis=1), use_container_width=True)

    # Grouped bar chart
    st.subheader("Visual Comparison")
    model_filter = st.multiselect(
        "Select models to compare:",
        options=model_order,
        default=["last_click", "markov", "shapley"],
    )

    if model_filter:
        filtered = all_results[all_results["model"].isin(model_filter)]
        fig_compare = px.bar(
            filtered,
            x="channel",
            y="revenue_pct",
            color="model",
            barmode="group",
            title="Attribution % by Channel and Model",
            labels={"revenue_pct": "Revenue Share (%)", "channel": "Channel"},
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_compare.update_layout(
            xaxis_tickangle=-45,
            template="plotly_white",
            height=500,
        )
        st.plotly_chart(fig_compare, use_container_width=True)

    # Key insight
    st.subheader("Key Insight: The Last-Click Bias Problem")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Last-Click Attribution**")
        lc = all_results[all_results["model"] == "last_click"].sort_values("revenue_pct", ascending=False)
        fig_lc = px.pie(lc, values="revenue_pct", names="channel",
                        title="Last-Click: Brand search dominates",
                        color_discrete_sequence=px.colors.qualitative.Set3)
        fig_lc.update_layout(height=400)
        st.plotly_chart(fig_lc, use_container_width=True)

    with col2:
        st.markdown("**Markov (Removal Effect) Attribution**")
        mk = all_results[all_results["model"] == "markov"].sort_values("revenue_pct", ascending=False)
        fig_mk = px.pie(mk, values="revenue_pct", names="channel",
                        title="Markov: Credit distributed more fairly",
                        color_discrete_sequence=px.colors.qualitative.Set3)
        fig_mk.update_layout(height=400)
        st.plotly_chart(fig_mk, use_container_width=True)

    # Markov removal effects
    st.subheader("Markov Removal Effect Analysis")
    st.markdown(
        "Shows the % drop in conversions if each channel were removed entirely. "
        "Channels with high removal effect are critical to the journey even if "
        "they rarely appear as the last click."
    )
    re_df = markov_results[["channel", "removal_effect"]].sort_values("removal_effect", ascending=True)
    fig_re = px.bar(
        re_df,
        x="removal_effect",
        y="channel",
        orientation="h",
        title="Channel Removal Effect (higher = more critical)",
        labels={"removal_effect": "Removal Effect", "channel": ""},
        color="removal_effect",
        color_continuous_scale="Reds",
    )
    fig_re.update_layout(template="plotly_white", height=450)
    st.plotly_chart(fig_re, use_container_width=True)


# =============================================================================
# Tab 3 — Booking Window Deep-Dive
# =============================================================================
with tab3:
    st.header("Booking Window Deep-Dive")
    st.markdown(
        "How does the channel mix differ by **booking lead time**? "
        "Last-minute and long-lead travellers have very different journeys."
    )

    # Segment summary
    summary = segment_summary(df)

    col1, col2, col3, col4 = st.columns(4)
    for i, (_, row) in enumerate(summary.iterrows()):
        with [col1, col2, col3, col4][i]:
            st.metric(
                row["segment_label"],
                f"{int(row['n_journeys'])} bookings",
                f"Avg {row['avg_touchpoints']:.1f} touchpoints",
            )

    # Stacked bar
    fig_mix = plot_channel_mix_by_segment(df)
    st.plotly_chart(fig_mix, use_container_width=True)

    # Touchpoint comparison
    fig_tp = plot_touchpoint_comparison(df)
    st.plotly_chart(fig_tp, use_container_width=True)

    # Detailed table
    st.subheader("Channel × Segment Breakdown")
    attr = attribution_by_segment(df)
    pivot_seg = attr.pivot_table(
        index="channel",
        columns="lead_time_segment",
        values="revenue_pct",
        aggfunc="first",
        fill_value=0,
    ).reindex(columns=SEGMENT_ORDER).round(1)
    pivot_seg.columns = [SEGMENT_LABELS.get(s, s) for s in pivot_seg.columns]
    st.dataframe(pivot_seg.style.background_gradient(cmap="Blues", axis=0), use_container_width=True)

    st.info(
        "**Key Finding:** Last-minute bookers convert via direct/brand search with "
        "minimal touchpoints. Long-lead bookers need 3× more social + metasearch "
        "touchpoints before converting."
    )


# =============================================================================
# Tab 4 — Budget Simulator
# =============================================================================
with tab4:
    st.header("Budget Simulator")
    st.markdown(
        "Drag the sliders to reallocate budget across channels and see the "
        "**projected impact on booking revenue**."
    )

    # Get baseline allocation (Markov-based)
    markov_results_cached = compute_markov(df)
    channels = sorted(df["channel"].unique())

    # Current "spend" estimated from last-click
    converting_ids = df.loc[df["converted"] == 1, "journey_id"].unique()
    last_touch_df = (
        df[df["journey_id"].isin(converting_ids)]
        .sort_values("session_timestamp")
        .groupby("journey_id")
        .last()
        .reset_index()
    )
    current_spend = last_touch_df.groupby("channel")["booking_value"].sum()
    current_spend_pct = (current_spend / current_spend.sum() * 100).to_dict()

    # Markov-recommended allocation
    markov_pct = markov_results_cached.set_index("channel")["revenue_pct"].to_dict()

    st.subheader("Adjust Channel Budget Allocation (%)")

    col_left, col_right = st.columns([1, 1])

    budget_alloc = {}
    half = len(channels) // 2

    with col_left:
        for ch in channels[:half]:
            default = int(round(current_spend_pct.get(ch, 10)))
            budget_alloc[ch] = st.slider(
                ch.replace("_", " ").title(),
                min_value=0, max_value=40,
                value=min(default, 40),
                key=f"budget_{ch}",
            )

    with col_right:
        for ch in channels[half:]:
            default = int(round(current_spend_pct.get(ch, 10)))
            budget_alloc[ch] = st.slider(
                ch.replace("_", " ").title(),
                min_value=0, max_value=40,
                value=min(default, 40),
                key=f"budget_{ch}",
            )

    total_alloc = sum(budget_alloc.values())
    if total_alloc == 0:
        total_alloc = 1

    # Normalise to 100%
    norm_alloc = {ch: v / total_alloc * 100 for ch, v in budget_alloc.items()}

    # Predicted revenue: use Markov elasticities as a simple model
    # Revenue = baseline × Σ(channel_weight × budget_change_factor)
    baseline_revenue = total_revenue

    predicted_factors = []
    for ch in channels:
        markov_weight = markov_pct.get(ch, 10) / 100
        current_pct = current_spend_pct.get(ch, 10)
        new_pct = norm_alloc.get(ch, 10)

        if current_pct > 0:
            change_ratio = new_pct / current_pct
        else:
            change_ratio = 1.0

        # Diminishing returns: sqrt of change ratio
        effective_change = np.sqrt(change_ratio) if change_ratio > 0 else 0
        predicted_factors.append(markov_weight * effective_change)

    predicted_revenue = baseline_revenue * sum(predicted_factors) / sum(
        markov_pct.get(ch, 10) / 100 for ch in channels
    )
    revenue_change = predicted_revenue - baseline_revenue
    revenue_change_pct = revenue_change / baseline_revenue * 100

    # Results
    st.markdown("---")
    st.subheader("Projected Impact")

    col1, col2, col3 = st.columns(3)
    col1.metric("Current Revenue", f"£{baseline_revenue:,.0f}")
    col2.metric("Predicted Revenue", f"£{predicted_revenue:,.0f}",
                f"{revenue_change_pct:+.1f}%")
    col3.metric("Revenue Change", f"£{revenue_change:+,.0f}")

    # Comparison chart
    comparison_data = pd.DataFrame({
        "channel": channels,
        "Current Allocation (%)": [current_spend_pct.get(ch, 0) for ch in channels],
        "Your Allocation (%)": [norm_alloc.get(ch, 0) for ch in channels],
        "Markov Recommended (%)": [markov_pct.get(ch, 0) for ch in channels],
    })

    fig_budget = go.Figure()
    fig_budget.add_trace(go.Bar(
        x=comparison_data["channel"],
        y=comparison_data["Current Allocation (%)"],
        name="Current (Last-Click)",
        marker_color="#e74c3c",
    ))
    fig_budget.add_trace(go.Bar(
        x=comparison_data["channel"],
        y=comparison_data["Your Allocation (%)"],
        name="Your Allocation",
        marker_color="#3498db",
    ))
    fig_budget.add_trace(go.Bar(
        x=comparison_data["channel"],
        y=comparison_data["Markov Recommended (%)"],
        name="Markov Recommended",
        marker_color="#2ecc71",
    ))

    fig_budget.update_layout(
        barmode="group",
        title="Budget Allocation Comparison",
        xaxis_tickangle=-45,
        yaxis_title="% of Budget",
        template="plotly_white",
        height=500,
    )
    st.plotly_chart(fig_budget, use_container_width=True)

    # Recommendation table
    st.subheader("Budget Recommendations")
    recs = budget_recommendation(df)
    st.dataframe(
        recs[["channel", "current_pct", "recommended_pct", "delta_pct", "recommendation"]]
        .round(1)
        .rename(columns={
            "channel": "Channel",
            "current_pct": "Current %",
            "recommended_pct": "Recommended %",
            "delta_pct": "Change (pp)",
            "recommendation": "Action",
        }),
        use_container_width=True,
        hide_index=True,
    )


# =============================================================================
# Tab 5 — Device Journey
# =============================================================================
with tab5:
    st.header("Cross-Device Journey Analysis")
    st.markdown(
        "Travel research commonly starts on **mobile** (inspiration) and "
        "ends on **desktop** (booking). This tab explores cross-device patterns."
    )

    # Cross-device stats
    cd = cross_device_analysis(df)
    pct_cross = cd["is_cross_device"].mean() * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Cross-Device Journeys", f"{pct_cross:.0f}%")
    col2.metric("Avg Devices per Journey", f"{cd['n_devices'].mean():.1f}")
    col3.metric("Most Common Start Device",
                cd["first_device"].mode().iloc[0] if len(cd) > 0 else "N/A")

    # Device transition heatmap
    st.subheader("Device Transition: First → Last")
    device_transitions = (
        cd.groupby(["first_device", "last_device"])
        .size()
        .reset_index(name="count")
    )
    pivot_dev = device_transitions.pivot_table(
        index="first_device",
        columns="last_device",
        values="count",
        fill_value=0,
    )

    fig_heat = px.imshow(
        pivot_dev.values,
        labels=dict(x="Last Device (Booking)", y="First Device (Discovery)", color="Journeys"),
        x=list(pivot_dev.columns),
        y=list(pivot_dev.index),
        color_continuous_scale="Blues",
        text_auto=True,
    )
    fig_heat.update_layout(title="Device Transitions in Converting Journeys", height=400)
    st.plotly_chart(fig_heat, use_container_width=True)

    # Channel by device
    st.subheader("Channel Usage by Device")

    converting_ids_set = set(df.loc[df["converted"] == 1, "journey_id"].unique())
    conv_df_full = df[df["journey_id"].isin(converting_ids_set)]

    ch_device = (
        conv_df_full.groupby(["channel", "device"])
        .size()
        .reset_index(name="touchpoints")
    )
    ch_device_total = ch_device.groupby("channel")["touchpoints"].transform("sum")
    ch_device["pct"] = ch_device["touchpoints"] / ch_device_total * 100

    fig_ch_dev = px.bar(
        ch_device,
        x="channel",
        y="pct",
        color="device",
        barmode="stack",
        title="Device Distribution by Channel (Converting Journeys)",
        labels={"pct": "% of Touchpoints", "channel": "Channel"},
        color_discrete_map={
            "mobile": "#e91e63",
            "tablet": "#ff9800",
            "desktop": "#3498db",
        },
    )
    fig_ch_dev.update_layout(
        xaxis_tickangle=-45,
        template="plotly_white",
        height=500,
    )
    st.plotly_chart(fig_ch_dev, use_container_width=True)

    # Top cross-device paths
    st.subheader("Top Cross-Device Journey Patterns")
    cross_only = cd[cd["is_cross_device"]].copy()
    if len(cross_only) > 0:
        # Simplify device paths for readability
        cross_only["simple_path"] = cross_only["devices"].apply(
            lambda devs: " → ".join(dict.fromkeys(devs))  # dedupe consecutive same
        )
        top_paths = cross_only["simple_path"].value_counts().head(10).reset_index()
        top_paths.columns = ["Device Path", "Count"]
        top_paths["% of Cross-Device"] = (top_paths["Count"] / len(cross_only) * 100).round(1)
        st.dataframe(top_paths, use_container_width=True, hide_index=True)

    st.info(
        "**Insight:** Mobile dominates early-funnel inspiration (social, blogs) "
        "while desktop dominates the booking moment (paid search, direct). "
        "Cross-device stitching via customer_id captures this full journey."
    )
