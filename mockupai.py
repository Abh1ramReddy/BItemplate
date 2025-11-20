import os
import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from openai import OpenAI
import streamlit as st

# -----------------------------
# OpenAI client
# -----------------------------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")

# -----------------------------
# Page config & basic styling
# -----------------------------
st.set_page_config(
    page_title="Kemb Reporting Mockup Generator",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .main {
            padding: 1.5rem 2rem;
        }
        h1, h2, h3 {
            font-weight: 700;
        }
        .section-header {
            font-size: 1.1rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 0.4rem;
            color: #0f4c81;
        }
        .kemb-card {
            border-radius: 18px;
            border: 1px solid rgba(15, 76, 129, 0.15);
            padding: 1rem 1.2rem;
            margin-bottom: 0.8rem;
            box-shadow: 0 8px 18px rgba(15, 76, 129, 0.04);
            background-color: white;
        }
        .pill {
            display: inline-block;
            padding: 0.25rem 0.8rem;
            border-radius: 999px;
            border: 1px solid rgba(15, 76, 129, 0.25);
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #0f4c81;
            background: rgba(15, 76, 129, 0.03);
            margin-right: 0.35rem;
            margin-bottom: 0.2rem;
        }
        .stDownloadButton button {
            border-radius: 999px !important;
            border: 1px solid rgba(15, 76, 129, 0.25) !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Default config (fallback)
# -----------------------------
DEFAULT_CONFIG = {
    "use_case_name": "Generic Performance Reporting",
    "primary_metric": "Primary KPI",
    "secondary_metrics": ["Secondary KPI 1", "Secondary KPI 2"],
    "dimension": "Segment",
    "segments": ["Segment A", "Segment B"],
    "base": 100.0,
    "charts": [
        {
            "title": "Primary KPI over time",
            "type": "line",          # line | area | bar | ... (see below)
            "level": "time_series",  # time_series | segment_latest | segment_over_time
            "y": ["Primary KPI"],
            "description": "High-level evolution of the main KPI."
        },
        {
            "title": "Primary KPI by segment (latest)",
            "type": "bar",
            "level": "segment_latest",
            "y": ["Primary KPI"],
            "description": "Distribution of performance across segments in the latest period."
        }
    ],
}
# Make sure default chart y uses the primary metric label
DEFAULT_CONFIG["charts"][0]["y"] = [DEFAULT_CONFIG["primary_metric"]]
DEFAULT_CONFIG["charts"][1]["y"] = [DEFAULT_CONFIG["primary_metric"]]


# -----------------------------
# Helpers
# -----------------------------
def generate_config_from_prompt(prompt: str) -> dict:
    """
    Use an LLM to turn a free-text prompt into a structured config.
    The prompt should describe: use case, metrics, dimension, segments, base AND desired charts.
    """
    if not prompt.strip():
        return DEFAULT_CONFIG

    if client is None:
        st.warning("OPENAI_API_KEY is not set. Using default configuration.")
        return DEFAULT_CONFIG

    system_message = (
        "You are a senior analytics consultant at a data-driven marketing/BI agency. "
        "You receive a description of a client's reporting needs and must produce a JSON config "
        "for a dashboard mockup. Use realistic metric names and segments for the described use case.\n\n"
        "Return ONLY valid JSON, no explanation."
    )

    user_instruction = f"""
    From the following description, extract a config for a reporting dashboard.

    Return JSON with exactly these keys:
    - use_case_name: short human-readable name of the use case
    - primary_metric: name of the main KPI (string)
    - secondary_metrics: array of 2-4 additional KPIs (strings). Include '%' in the name for percentage metrics where appropriate.
    - dimension: the dimension used for segment breakdown (e.g. "Channel", "Plan", "Country")
    - segments: array of 3-6 segment values for that dimension (strings)
    - base: a positive number indicating the rough magnitude of the primary metric per period 
            (e.g. 200 for leads, 15000 for revenue, 2000 for sessions)

    Additionally define a list of chart specifications under key "charts".
    - charts: array of chart objects, each with:
        - title: short title for the chart (string)
        - type: one of:
            "line", "area", "bar", "bar_horizontal",
            "stacked_area", "stacked_bar",
            "pie", "donut",
            "scatter", "heatmap"
        - level: one of:
            - "time_series" -> use time-series data with Date on x-axis
            - "segment_latest" -> use the latest period, aggregated by the dimension
            - "segment_over_time" -> use segment mix over time (pivot: Date x segment, based on the primary metric)
        - y: array of 1-3 metric names to plot (strings). For segment charts, these refer to metrics to aggregate.
        - description: 1–2 sentences about the purpose of the chart

    Guidelines:
    - Include 2–5 charts.
    - At least one should be a time-series chart of the primary metric.
    - At least one should be a segment breakdown (latest or over time).
    - Use only the metrics and segments defined above; do not invent new metrics or segments.
    - Prefer:
        - pie/donut only for 'segment_latest'
        - stacked_area/stacked_bar only for 'segment_over_time'
        - scatter/line/area primarily for 'time_series'

    Client description:
    \"\"\"{prompt}\"\"\"
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_instruction},
            ],
        )
        content = response.choices[0].message.content
        cfg = json.loads(content)

        # basic validation / fallback defaults
        for key, val in DEFAULT_CONFIG.items():
            cfg.setdefault(key, val)

        # ensure list types
        if not isinstance(cfg.get("secondary_metrics"), list):
            cfg["secondary_metrics"] = DEFAULT_CONFIG["secondary_metrics"]
        if not isinstance(cfg.get("segments"), list):
            cfg["segments"] = DEFAULT_CONFIG["segments"]
        if not isinstance(cfg.get("charts"), list) or len(cfg["charts"]) == 0:
            cfg["charts"] = DEFAULT_CONFIG["charts"]
        if not isinstance(cfg.get("base"), (int, float)):
            cfg["base"] = DEFAULT_CONFIG["base"]

        # Normalize chart y metric names if they used generic placeholders
        for chart in cfg["charts"]:
            if "y" in chart and isinstance(chart["y"], list):
                chart["y"] = [
                    (m.replace("Primary KPI", cfg["primary_metric"]))
                    for m in chart["y"]
                ]

        return cfg

    except Exception as e:
        st.warning(f"Could not interpret prompt with AI. Using default config. (Error: {e})")
        return DEFAULT_CONFIG


def generate_time_index(periods: int, frequency: str):
    freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "MS"}
    freq = freq_map.get(frequency, "W")
    end = date.today()
    start = end - timedelta(days=periods * 7) if frequency == "Weekly" else end - timedelta(days=periods * 30)
    return pd.date_range(start=start, periods=periods, freq=freq)


def generate_dummy_data(cfg: dict, periods: int, frequency: str, noise_level: float = 0.2):
    idx = generate_time_index(periods, frequency)
    base = float(cfg.get("base", 100.0))
    primary_metric = cfg["primary_metric"]
    secondary_metrics = cfg["secondary_metrics"]
    dim = cfg["dimension"]
    segments = cfg["segments"]

    # Primary metric trend + noise
    trend = np.linspace(0.9, 1.1, periods)
    noise = np.random.normal(loc=1.0, scale=noise_level, size=periods)
    primary_values = (base * trend * noise).clip(min=0)

    df = pd.DataFrame({"Date": idx, primary_metric: primary_values})

    # Secondary metrics derived from primary
    for metric in secondary_metrics:
        if "%" in metric:
            df[metric] = (np.random.uniform(1, 25, size=periods)).round(2)
        else:
            factor = np.random.uniform(0.1, 1.2)
            df[metric] = (df[primary_metric] * factor).round(0)

    # Segment-level breakdown (only primary metric for now)
    segment_records = []
    for d in idx:
        base_value = base * np.random.uniform(0.6, 1.4)
        weights = np.random.dirichlet(np.ones(len(segments)))
        for seg, w in zip(segments, weights):
            segment_records.append(
                {
                    "Date": d,
                    dim: seg,
                    primary_metric: max(base_value * w * np.random.uniform(0.7, 1.3), 0),
                }
            )

    df_segments = pd.DataFrame(segment_records)
    return df, df_segments


def compute_kpi_change(series: pd.Series):
    if len(series) < 2:
        return series.iloc[-1], 0.0
    current = series.iloc[-1]
    previous = series.iloc[-2]
    if previous == 0:
        pct_change = 0.0
    else:
        pct_change = ((current - previous) / abs(previous)) * 100
    return current, pct_change


def render_dynamic_views(cfg: dict, df_time: pd.DataFrame, df_segments: pd.DataFrame):
    """
    Render charts based on the 'charts' spec in cfg.

    Supported:
      - level == "time_series": use df_time (Date on x)
      - level == "segment_latest": aggregate df_segments for latest Date by dimension
      - level == "segment_over_time": pivot df_segments into Date x dimension (primary metric)
    Chart types:
      - line, area, bar
      - bar_horizontal
      - stacked_area, stacked_bar
      - pie, donut
      - scatter
      - heatmap
    """
    primary = cfg["primary_metric"]
    dim = cfg["dimension"]
    charts = cfg.get("charts", [])

    if not charts:
        st.info("No charts defined in configuration. Add 'charts' to the prompt or config.")
        return

    # Precompute segment-based structures once
    latest_date = df_segments["Date"].max() if not df_segments.empty else None
    seg_latest = pd.DataFrame()
    if latest_date is not None:
        seg_latest = (
            df_segments[df_segments["Date"] == latest_date]
            .groupby(dim)
            .sum(numeric_only=True)
        )

    seg_over_time = pd.DataFrame()
    if not df_segments.empty:
        seg_over_time = df_segments.pivot_table(
            index="Date",
            columns=dim,
            values=primary,
            aggfunc="sum",
        ).fillna(0)

    st.markdown('<div class="section-header">Visualizations</div>', unsafe_allow_html=True)

    for chart in charts:
        title = chart.get("title", "Chart")
        ctype = chart.get("type", "line")  # chart type
        level = chart.get("level", "time_series")
        y_metrics = chart.get("y", [primary])
        description = chart.get("description", "")

        with st.container():
            st.markdown('<div class="kemb-card">', unsafe_allow_html=True)
            st.subheader(title)

            # TIME SERIES LEVEL
            if level == "time_series":
                base_df = df_time.set_index("Date")
                cols = [m for m in y_metrics if m in base_df.columns]
                if not cols:
                    st.warning("No matching metrics found in time-series for this chart.")
                else:
                    plot_df = base_df[cols]

                    if ctype in ["line", "area", "bar"]:
                        # Use Streamlit native charts
                        if ctype == "line":
                            st.line_chart(plot_df)
                        elif ctype == "area":
                            st.area_chart(plot_df)
                        else:
                            st.bar_chart(plot_df)

                    elif ctype == "scatter":
                        # Use first metric for scatter against Date
                        metric = cols[0]
                        fig = px.scatter(
                            plot_df.reset_index(),
                            x="Date",
                            y=metric,
                            title=None,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    elif ctype == "heatmap":
                        # Heatmap of metrics over time
                        fig = px.imshow(
                            plot_df.T,
                            aspect="auto",
                            labels=dict(x="Date", y="Metric", color="Value"),
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    else:
                        # Fallback to line
                        st.line_chart(plot_df)

            # SEGMENT LATEST LEVEL
            elif level == "segment_latest":
                if seg_latest.empty:
                    st.warning("No segment data available for latest period.")
                else:
                    available_cols = [m for m in y_metrics if m in seg_latest.columns]
                    if not available_cols:
                        if primary in seg_latest.columns:
                            available_cols = [primary]
                        else:
                            available_cols = seg_latest.columns[:1].tolist()

                    plot_df = seg_latest[available_cols]

                    if ctype == "bar":
                        # Vertical bar via Plotly for more control
                        fig = px.bar(
                            plot_df.reset_index(),
                            x=dim,
                            y=available_cols[0] if len(available_cols) == 1 else available_cols,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    elif ctype == "bar_horizontal":
                        fig = px.bar(
                            plot_df.reset_index(),
                            x=available_cols[0] if len(available_cols) == 1 else available_cols,
                            y=dim,
                            orientation="h",
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    elif ctype in ["pie", "donut"]:
                        # Use first metric for share
                        metric = available_cols[0]
                        fig = px.pie(
                            plot_df.reset_index(),
                            names=dim,
                            values=metric,
                            hole=0.5 if ctype == "donut" else 0.0,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    elif ctype == "heatmap":
                        fig = px.imshow(
                            plot_df.T,
                            aspect="auto",
                            labels=dict(x="Segment", y="Metric", color="Value"),
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    else:
                        # default: vertical bar for latest
                        fig = px.bar(
                            plot_df.reset_index(),
                            x=dim,
                            y=available_cols[0] if len(available_cols) == 1 else available_cols,
                        )
                        st.plotly_chart(fig, use_container_width=True)

            # SEGMENT OVER TIME LEVEL
            elif level == "segment_over_time":
                if seg_over_time.empty:
                    st.warning("No segment-over-time data available.")
                else:
                    # seg_over_time: index=Date, columns=segments, values=primary metric
                    df_plot = seg_over_time.copy()
                    df_melt = df_plot.reset_index().melt(
                        id_vars="Date", var_name=dim, value_name=primary
                    )

                    if ctype in ["stacked_area", "area"]:
                        fig = px.area(
                            df_melt,
                            x="Date",
                            y=primary,
                            color=dim,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    elif ctype in ["stacked_bar", "bar"]:
                        fig = px.bar(
                            df_melt,
                            x="Date",
                            y=primary,
                            color=dim,
                            barmode="stack",
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    elif ctype == "heatmap":
                        fig = px.imshow(
                            df_plot.T,
                            aspect="auto",
                            labels=dict(x="Date", y=dim, color=primary),
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    elif ctype in ["pie", "donut"]:
                        # Aggregate over full time for total contribution
                        total_by_segment = df_plot.sum(axis=0)
                        fig = px.pie(
                            total_by_segment.reset_index(),
                            names=total_by_segment.index.name or dim,
                            values=0,
                            hole=0.5 if ctype == "donut" else 0.0,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    else:
                        # Fallback to stacked area
                        fig = px.area(
                            df_melt,
                            x="Date",
                            y=primary,
                            color=dim,
                        )
                        st.plotly_chart(fig, use_container_width=True)

            else:
                st.warning(f"Unknown chart level: {level}")

            if description:
                st.caption(description)

            st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# Sidebar – configuration
# -----------------------------
st.sidebar.title("Kemb Mockup Settings")

prompt = st.sidebar.text_area(
    "Reporting configuration prompt",
    placeholder=(
        "Example:\n"
        "We are a D2C cosmetics brand. We care about Revenue as primary metric.\n"
        "Secondary metrics: Sessions, Conversion Rate %, Average Order Value €.\n"
        "We break things down by Channel with segments: Organic Search, Paid Search, Paid Social, Direct, Email.\n"
        "Typical weekly revenue is around 25,000 €.\n\n"
        "Charts:\n"
        "- Line chart of Revenue and AOV over time (time_series).\n"
        "- Donut chart of Revenue by Channel for latest period (segment_latest).\n"
        "- Stacked area chart of Revenue mix by Channel over time (segment_over_time).\n"
        "- Scatter of Revenue vs Conversion Rate % over time (time_series).\n"
        "- Heatmap of metrics over time (time_series)."
    ),
    height=260,
)

frequency = st.sidebar.radio("Time granularity", options=["Weekly", "Monthly", "Daily"], index=1)

periods = st.sidebar.slider(
    "Number of periods",
    min_value=6,
    max_value=24,
    value=12,
    help="How long the dummy time series should be.",
)

noise_level = st.sidebar.slider(
    "Data volatility",
    min_value=0.05,
    max_value=0.5,
    value=0.2,
    step=0.05,
    help="Higher values create 'messier' data.",
)

st.sidebar.markdown("---")
interpret_btn = st.sidebar.button("Interpret prompt (AI)")
generate_btn = st.sidebar.button("Generate mockup with current config")

# Keep config in session so you can iterate without re-calling the model every time
if "dashboard_config" not in st.session_state:
    st.session_state["dashboard_config"] = DEFAULT_CONFIG

if interpret_btn:
    cfg = generate_config_from_prompt(prompt)
    st.session_state["dashboard_config"] = cfg

cfg = st.session_state["dashboard_config"]

# -----------------------------
# Main layout
# -----------------------------
st.markdown("# Kemb Reporting Mockup Generator")

col_info, col_meta = st.columns([2.5, 1.5])

with col_info:
    st.markdown('<div class="section-header">Client / use case description</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="kemb-card">', unsafe_allow_html=True)
        if prompt.strip():
            st.write(prompt)
        else:
            st.write(
                "Use the sidebar prompt to describe the client, use case, metrics, segments, "
                "and the charts you want to see. Then click **Interpret prompt (AI)**."
            )
        st.markdown("</div>", unsafe_allow_html=True)

with col_meta:
    st.markdown('<div class="section-header">Current configuration</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="kemb-card">', unsafe_allow_html=True)
        st.write(f"**Use case:** {cfg['use_case_name']}")
        st.write(f"**Primary KPI:** `{cfg['primary_metric']}`")

        st.write("Secondary KPIs:")
        if cfg["secondary_metrics"]:
            pills = " ".join([f'<span class="pill">{m}</span>' for m in cfg["secondary_metrics"]])
            st.markdown(pills, unsafe_allow_html=True)
        else:
            st.write("_None defined_")

        st.write(f"**Dimension:** `{cfg['dimension']}`")
        st.write("Segments:")
        seg_pills = " ".join([f'<span class="pill">{s}</span>' for s in cfg["segments"]])
        st.markdown(seg_pills, unsafe_allow_html=True)

        st.write(f"**Base magnitude:** ~{cfg['base']}")
        st.caption("Base defines the rough scale of the primary metric per period for dummy data.")

        st.write("Charts:")
        for c in cfg.get("charts", []):
            st.markdown(
                f"- **{c.get('title', 'Chart')}** "
                f"(_type_: `{c.get('type', 'line')}`, _level_: `{c.get('level', 'time_series')}`)"
            )

        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

if generate_btn:
    # Data generation
    df_time, df_segments = generate_dummy_data(cfg, periods, frequency, noise_level)
    primary_metric = cfg["primary_metric"]

    # KPI tiles
    st.markdown('<div class="section-header">Key metrics</div>', unsafe_allow_html=True)
    sec_metrics = cfg["secondary_metrics"]
    kpi_cols = st.columns(1 + len(sec_metrics))

    current_primary, primary_change = compute_kpi_change(df_time[primary_metric])
    with kpi_cols[0]:
        st.metric(
            label=primary_metric,
            value=f"{current_primary:,.0f}",
            delta=f"{primary_change:+.1f}%",
        )

    for col, metric in zip(kpi_cols[1:], sec_metrics):
        current, change = compute_kpi_change(df_time[metric])
        fmt = "{:,.0f}" if "%" not in metric else "{:,.2f}"
        with col:
            st.metric(
                label=metric,
                value=fmt.format(current),
                delta=f"{change:+.1f}%",
            )

    st.markdown("---")

    # Dynamic views based on config + chart spec
    render_dynamic_views(cfg, df_time, df_segments)

    # Data preview
    st.markdown('<div class="section-header">Data preview</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="kemb-card">', unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["Time-series KPIs", "Segment breakdown"])
        with tab1:
            st.dataframe(df_time.tail(10), use_container_width=True)
        with tab2:
            st.dataframe(df_segments.tail(20), use_container_width=True)

        csv = df_time.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download time-series data as CSV",
            data=csv,
            file_name="kemb_mockup_timeseries.csv",
            mime="text/csv",
        )
        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info(
        "1) Describe the reporting setup (metrics, segments, chart types) in the sidebar prompt.\n"
        "2) Click **Interpret prompt (AI)**.\n"
        "3) Then click **Generate mockup with current config**."
    )

