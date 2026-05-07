# app.py — Travel Mex Tours | Social Media Analytics Dashboard v3
# Updated to use Performance Score as primary metric — corrects reach bias

import sys, warnings, io
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from load_data import load_instagram_csv
from preprocessing import preprocess_df

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Travel Mex | Analytics",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BENCHMARK_ER  = 5.0    # industry benchmark for 1k-10k follower accounts
WEEKDAY_ORDER = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .kpi-card {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 14px;
        padding: 20px 22px;
        margin-bottom: 4px;
    }
    .kpi-label {
        font-size: 12px; font-weight: 600;
        letter-spacing: 0.08em; text-transform: uppercase;
        color: rgba(255,255,255,0.55); margin-bottom: 6px;
    }
    .kpi-value { font-size: 32px; font-weight: 700; color: #ffffff; line-height: 1; }
    .kpi-delta { font-size: 12px; margin-top: 6px; font-weight: 500; }
    .kpi-delta.up   { color: #4ade80; }
    .kpi-delta.down { color: #f87171; }
    .kpi-delta.info { color: #93c5fd; }

    .section-title {
        font-size: 13px; font-weight: 700;
        letter-spacing: 0.1em; text-transform: uppercase;
        color: rgba(255,255,255,0.4); margin: 28px 0 14px 2px;
    }
    .alert { padding: 14px 18px; border-radius: 10px; font-size: 14px; font-weight: 500; margin: 12px 0; line-height: 1.5; }
    .alert-green  { background: rgba(74,222,128,0.12); border: 1px solid rgba(74,222,128,0.25); color: #86efac; }
    .alert-yellow { background: rgba(251,191,36,0.12); border: 1px solid rgba(251,191,36,0.25); color: #fde68a; }
    .alert-blue   { background: rgba(96,165,250,0.12); border: 1px solid rgba(96,165,250,0.25); color: #bfdbfe; }
    .alert-purple { background: rgba(167,139,250,0.12); border: 1px solid rgba(167,139,250,0.25); color: #ddd6fe; }

    .predict-box {
        background: linear-gradient(135deg, rgba(24,119,242,0.25) 0%, rgba(131,58,180,0.25) 100%);
        border: 1px solid rgba(131,58,180,0.4);
        border-radius: 16px; padding: 24px; text-align: center;
    }
    .predict-number { font-size: 48px; font-weight: 700; color: #ffffff; line-height: 1; }
    .predict-label  { font-size: 13px; color: rgba(255,255,255,0.6); margin-top: 6px; }

    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 24px !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_data(raw: bytes):
    df_raw   = load_instagram_csv(io.BytesIO(raw))
    df, meta = preprocess_df(df_raw)
    return df, meta


with st.sidebar:
    st.markdown("### ✈️ Travel Mex Tours")
    st.markdown("---")
    uploaded = st.file_uploader("Upload Instagram CSV", type=["csv"])
    st.markdown("---")
    benchmark_er = st.slider("ER Benchmark (%)", 1.0, 15.0, BENCHMARK_ER, 0.5,
                              help="Industry avg for 1k-10k accounts: 3-6%")
    st.caption("👥 Followers: 2,608  •  Instagram")

df, meta = None, None
if uploaded:
    try:
        raw = uploaded.read()
        df, meta = load_data(raw)
        st.sidebar.success(f"✅ {uploaded.name}")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")
        st.stop()
else:
    data_dir  = Path(__file__).parent / "data" / "instagram"
    csv_files = sorted(data_dir.glob("*.csv"), key=lambda f: f.stat().st_mtime, reverse=True)
    if csv_files:
        raw = csv_files[0].read_bytes()
        df, meta = load_data(raw)
        st.sidebar.info(f"📁 {csv_files[0].name}")

if df is None:
    st.markdown("""
    <div style="text-align:center;padding:80px 20px;">
        <div style="font-size:64px">✈️</div>
        <h2 style="color:white">Travel Mex Analytics</h2>
        <p style="color:rgba(255,255,255,0.5)">Upload your Instagram CSV from the sidebar</p>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ── Shortcuts ──────────────────────────────────────────────────────────────
er_col        = meta["engagement_rate_col"]
ps_col        = meta["performance_score_col"]
tier_col      = meta["performance_tier_col"]
tier_rel      = meta["performance_tier_relative_col"]
date_col      = meta.get("date_col")
type_col      = meta.get("type_col")
time_slot_col = meta.get("time_slot_col")
likes_col     = meta.get("likes_col")
saves_col     = meta.get("saves_col")
comments_col  = meta.get("comments_col")
shares_col    = meta.get("shares_col")
reach_col     = meta.get("reach_col")

er  = df[er_col].dropna()
ps  = df[ps_col].dropna()
avg_er  = round(er.mean(), 2)
avg_ps  = round(ps.mean(), 2)

best_day  = df.groupby("weekday")[ps_col].mean().idxmax() if "weekday" in df.columns else "N/A"
best_hour = int(df.groupby("hour")[ps_col].mean().idxmax()) if "hour" in df.columns else None
best_type = df.groupby(type_col)[ps_col].mean().idxmax() if type_col and type_col in df.columns else "N/A"
best_slot = df.groupby(time_slot_col, observed=True)[ps_col].mean().idxmax() if time_slot_col and time_slot_col in df.columns else "N/A"


# ══════════════════════════════════════════════════════════════════════════
# NAVIGATION
# ══════════════════════════════════════════════════════════════════════════

tab_overview, tab_timing, tab_content, tab_predictor = st.tabs([
    "📊  Overview",
    "📅  Best Time to Post",
    "🎬  Content Performance",
    "🤖  Engagement Predictor"
])


# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════

with tab_overview:
    st.markdown(f"""
    <div style="margin-bottom:24px">
        <div style="font-size:26px;font-weight:700;color:white">Performance Overview</div>
        <div style="color:rgba(255,255,255,0.45);font-size:14px;margin-top:4px">
            {len(df)} posts · {meta.get("date_range_start","?")} → {meta.get("date_range_end","?")}
        </div>
    </div>""", unsafe_allow_html=True)

    # ── KPI cards ──────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)

    with k1:
        delta = round(avg_ps - 50, 1)
        cls   = "up" if delta >= 0 else "down"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Avg Performance Score</div>
            <div class="kpi-value">{avg_ps}<span style="font-size:16px">/100</span></div>
            <div class="kpi-delta {cls}">{'▲' if delta>=0 else '▼'} {abs(delta)} vs 50pt midpoint</div>
        </div>""", unsafe_allow_html=True)

    with k2:
        delta_er = round(avg_er - benchmark_er, 1)
        cls_er   = "up" if delta_er >= 0 else "down"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Avg Engagement Rate</div>
            <div class="kpi-value">{avg_er}<span style="font-size:16px">%</span></div>
            <div class="kpi-delta {cls_er}">{'▲' if delta_er>=0 else '▼'} {abs(delta_er)}% vs {benchmark_er}% benchmark</div>
        </div>""", unsafe_allow_html=True)

    with k3:
        high_n = meta.get("high_performers_count", 0)
        above_n = meta.get("above_avg_count", 0)
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">High Performers</div>
            <div class="kpi-value">{high_n}</div>
            <div class="kpi-delta info">{above_n} above avg (relative)</div>
        </div>""", unsafe_allow_html=True)

    with k4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Best Day to Post</div>
            <div class="kpi-value" style="font-size:22px">{best_day}</div>
            <div class="kpi-delta up">▲ by Performance Score</div>
        </div>""", unsafe_allow_html=True)

    with k5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Best Content Type</div>
            <div class="kpi-value" style="font-size:20px">{best_type}</div>
            <div class="kpi-delta up">▲ by Performance Score</div>
        </div>""", unsafe_allow_html=True)

    # ── Explanation banner ──────────────────────────────────────────────────
    st.markdown("""
    <div class="alert alert-purple">
        📊 <strong>Performance Score (0-100)</strong> is our primary metric.
        It combines Engagement Rate (40%) + Total Interactions (40%) + Reach (20%)
        to correct for reach bias — posts that reach more people are not unfairly penalized
        for having lower ER.
    </div>""", unsafe_allow_html=True)

    if avg_er >= benchmark_er * 1.5:
        st.markdown(f"""
        <div class="alert alert-green">
            ✅ Your {avg_er:.1f}% avg ER is <strong>{avg_er/benchmark_er:.1f}× above</strong>
            the {benchmark_er}% benchmark for 1k–10k follower accounts.
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # ── Charts ─────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["Performance Score Dist.",
                                            "Performance Tiers"])

        fig.add_trace(go.Histogram(
            x=ps, nbinsx=20,
            marker_color="rgba(96,165,250,0.7)",
            marker_line_color="rgba(96,165,250,1)",
            marker_line_width=1
        ), row=1, col=1)
        fig.add_vline(x=avg_ps, line_dash="dash", line_color="#4ade80",
                      annotation_text=f"Avg {avg_ps}", row=1, col=1)

        tier_counts = df[tier_col].value_counts().reindex(["High","Medium","Low"]).fillna(0)
        fig.add_trace(go.Bar(
            x=tier_counts.index, y=tier_counts.values,
            marker_color=["#4ade80","#fbbf24","#f87171"],
            text=tier_counts.values.astype(int), textposition="outside"
        ), row=1, col=2)

        fig.update_layout(
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            font_color="rgba(255,255,255,0.7)", height=300,
            margin=dict(t=40,b=10,l=10,r=10)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        breakdown = {k: v for k, v in {
            "Likes": likes_col, "Comments": comments_col,
            "Saves": saves_col, "Shares": shares_col,
        }.items() if v and v in df.columns}

        if breakdown:
            totals = {k: int(df[v].sum()) for k, v in breakdown.items()}
            fig2 = go.Figure(go.Pie(
                labels=list(totals.keys()), values=list(totals.values()),
                hole=0.5,
                marker_colors=["#60a5fa","#a78bfa","#f472b6","#fbbf24"],
                textinfo="percent", textfont_size=12
            ))
            fig2.update_layout(
                title="Interaction Breakdown",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="rgba(255,255,255,0.7)", height=300,
                margin=dict(t=40,b=0,l=0,r=0),
                legend=dict(font=dict(color="rgba(255,255,255,0.6)"))
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── Top 5 posts by Performance Score ───────────────────────────────────
    st.markdown('<div class="section-title">🏆 Top 5 Posts by Performance Score</div>',
                unsafe_allow_html=True)

    top5    = df.nlargest(5, ps_col)
    medals  = ["🥇","🥈","🥉","4️⃣","5️⃣"]

    for i, (_, row) in enumerate(top5.iterrows()):
        date_str = str(row[date_col].date()) if date_col and pd.notna(row.get(date_col)) else "N/A"
        type_str = str(row[type_col]) if type_col and pd.notna(row.get(type_col)) else ""
        eng_str  = f"{int(row['total_engagements'])} interactions" if pd.notna(row.get("total_engagements")) else ""
        reach_str = f"reach {int(row[reach_col])}" if reach_col and pd.notna(row.get(reach_col)) else ""

        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.08);
             border-radius:8px;padding:10px 16px;margin:4px 0;display:flex;align-items:center;">
            <div style="font-size:18px;font-weight:700;color:rgba(255,255,255,0.3);width:36px">{medals[i]}</div>
            <div style="color:rgba(255,255,255,0.85);flex:1">
                <span style="font-weight:600">{date_str}</span>
                {"&nbsp;·&nbsp;" + type_str if type_str else ""}
                {"&nbsp;·&nbsp;" + eng_str if eng_str else ""}
                {"&nbsp;·&nbsp;" + reach_str if reach_str else ""}
            </div>
            <div style="display:flex;gap:16px;align-items:center">
                <span style="color:#93c5fd;font-size:13px">ER {row[er_col]:.1f}%</span>
                <span style="font-size:20px;font-weight:700;color:#4ade80">
                    {row[ps_col]:.1f}<span style="font-size:12px">/100</span>
                </span>
            </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — BEST TIME
# ══════════════════════════════════════════════════════════════════════════

with tab_timing:
    st.markdown("""
    <div style="font-size:26px;font-weight:700;color:white;margin-bottom:6px">Best Time to Post</div>
    <div style="color:rgba(255,255,255,0.45);font-size:14px;margin-bottom:16px">
        Ranked by Performance Score — corrects for reach bias
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="alert alert-blue">
        ⏰ <strong>Note on time slots:</strong> We group hours into slots because 63% of posts
        are at 9:00 — comparing individual hours is not statistically valid.
        Slots give enough samples per group for meaningful comparisons.
    </div>""", unsafe_allow_html=True)

    if "weekday" not in df.columns:
        st.info("No date data found in your CSV.")
    else:
        col_l, col_r = st.columns(2)

        with col_l:
            day_ps = df.groupby("weekday")[ps_col].mean().reindex(
                [d for d in WEEKDAY_ORDER if d in df["weekday"].unique()]
            ).reset_index()
            day_ps.columns = ["Day", "Avg Score"]

            colors = ["#4ade80" if d == best_day else "#3b82f6" for d in day_ps["Day"]]
            fig = go.Figure(go.Bar(
                x=day_ps["Day"], y=day_ps["Avg Score"],
                marker_color=colors,
                text=day_ps["Avg Score"].round(1), textposition="outside",
                textfont=dict(color="white", size=10)
            ))
            fig.update_layout(
                title=f"Best day: <b>{best_day}</b> ⭐",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="rgba(255,255,255,0.7)",
                yaxis=dict(gridcolor="rgba(255,255,255,0.07)", title="Avg Performance Score"),
                margin=dict(t=50,b=20,l=10,r=10)
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            if time_slot_col and time_slot_col in df.columns:
                slot_ps = df.groupby(time_slot_col, observed=True)[ps_col].mean().reset_index()
                slot_ps.columns = ["Slot", "Avg Score"]
                slot_ps = slot_ps[slot_ps["Avg Score"] > 0]

                colors_s = ["#4ade80" if str(s) == str(best_slot) else "#a78bfa"
                            for s in slot_ps["Slot"]]
                fig2 = go.Figure(go.Bar(
                    x=slot_ps["Slot"].astype(str), y=slot_ps["Avg Score"],
                    marker_color=colors_s,
                    text=slot_ps["Avg Score"].round(1), textposition="outside",
                    textfont=dict(color="white", size=10)
                ))
                fig2.update_layout(
                    title=f"Best slot: <b>{best_slot}</b> ⭐",
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font_color="rgba(255,255,255,0.7)",
                    yaxis=dict(gridcolor="rgba(255,255,255,0.07)", title="Avg Performance Score"),
                    xaxis=dict(tickangle=-15),
                    margin=dict(t=50,b=20,l=10,r=10)
                )
                st.plotly_chart(fig2, use_container_width=True)

        st.markdown(f"""
        <div class="alert alert-green">
            📅 <strong>Recommendation:</strong> Post on <strong>{best_day}</strong>
            during <strong>{best_slot}</strong> for maximum Performance Score.
        </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div class="alert alert-yellow">
            ⚠️ <strong>Statistical note:</strong> Early Morning has ~89 posts vs ~16 in other slots.
            More data is needed in Late Morning and Afternoon before drawing firm conclusions.
            Run the recommended experiment: post 10x in each slot over 3 months.
        </div>""", unsafe_allow_html=True)

        # Weekend vs Weekday
        if "is_weekend" in df.columns:
            weekend_n = int((df["is_weekend"]==True).sum())
            if weekend_n < 5:
                st.markdown(f"""
                <div class="alert alert-yellow">
                    📅 <strong>Weekend vs Weekday:</strong> Only {weekend_n} weekend post(s) —
                    not enough for a valid comparison. Post on weekends for 2-3 months to unlock this analysis.
                </div>""", unsafe_allow_html=True)
            else:
                wk_df = df.copy()
                wk_df["Period"] = wk_df["is_weekend"].map({True:"Weekend 🌅", False:"Weekday 💼"})
                wk_avg = wk_df.groupby("Period")[ps_col].mean().reset_index()
                fig3 = go.Figure(go.Bar(
                    x=wk_avg["Period"], y=wk_avg[ps_col],
                    marker_color=["#a78bfa","#60a5fa"],
                    text=wk_avg[ps_col].round(1), textposition="outside",
                    textfont=dict(color="white")
                ))
                fig3.update_layout(
                    title="Weekend vs Weekday — Performance Score",
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font_color="rgba(255,255,255,0.7)",
                    yaxis=dict(gridcolor="rgba(255,255,255,0.07)"),
                    margin=dict(t=40,b=20,l=10,r=10), showlegend=False
                )
                st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — CONTENT PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════

with tab_content:
    st.markdown("""
    <div style="font-size:26px;font-weight:700;color:white;margin-bottom:6px">Content Performance</div>
    <div style="color:rgba(255,255,255,0.45);font-size:14px;margin-bottom:16px">
        Primary metric: Performance Score · Secondary: Engagement Rate
    </div>""", unsafe_allow_html=True)

    if not type_col or type_col not in df.columns:
        st.info("No content type column found.")
    else:
        type_stats = df.groupby(type_col).agg(
            Posts=(er_col, "count"),
            Avg_Score=(ps_col, "mean"),
            Avg_ER=(er_col, "mean"),
            Median_Score=(ps_col, "median")
        ).round(2).sort_values("Avg_Score", ascending=False).reset_index()

        col_l, col_r = st.columns([3, 2])

        with col_l:
            fig = make_subplots(rows=1, cols=2,
                                subplot_titles=["Performance Score", "Engagement Rate"])

            colors_ps = ["#4ade80" if t == best_type else "#3b82f6" for t in type_stats[type_col]]
            colors_er = ["#4ade80" if t == type_stats.loc[type_stats["Avg_ER"].idxmax(), type_col]
                         else "#a78bfa" for t in type_stats[type_col]]

            fig.add_trace(go.Bar(
                x=type_stats[type_col], y=type_stats["Avg_Score"],
                marker_color=colors_ps,
                text=type_stats["Avg_Score"].astype(str), textposition="outside"
            ), row=1, col=1)

            fig.add_trace(go.Bar(
                x=type_stats[type_col], y=type_stats["Avg_ER"],
                marker_color=colors_er,
                text=type_stats["Avg_ER"].astype(str)+"%", textposition="outside"
            ), row=1, col=2)

            fig.update_layout(
                showlegend=False,
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="rgba(255,255,255,0.7)",
                margin=dict(t=40,b=20,l=10,r=10)
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            for _, row in type_stats.iterrows():
                is_best = row[type_col] == best_type
                st.markdown(f"""
                <div style="background:rgba(255,255,255,{'0.08' if is_best else '0.04'});
                     border:1px solid rgba(255,255,255,{'0.2' if is_best else '0.08'});
                     border-radius:10px;padding:12px 16px;margin-bottom:8px;">
                    <div style="display:flex;justify-content:space-between;align-items:center">
                        <span style="font-weight:600;color:white">
                            {'⭐ ' if is_best else ''}{row[type_col]}
                        </span>
                        <span style="font-size:20px;font-weight:700;color:#4ade80">
                            {row['Avg_Score']}/100
                        </span>
                    </div>
                    <div style="color:rgba(255,255,255,0.4);font-size:12px;margin-top:4px">
                        {int(row['Posts'])} posts · ER {row['Avg_ER']}%
                    </div>
                </div>""", unsafe_allow_html=True)

        # Monthly trend
        if date_col and pd.api.types.is_datetime64_any_dtype(df[date_col]):
            monthly_ps = df.set_index(date_col)[ps_col].resample("ME").mean().dropna().reset_index()
            monthly_ps.columns = ["Month", "Avg Score"]
            monthly_er = df.set_index(date_col)[er_col].resample("ME").mean().dropna().reset_index()
            monthly_er.columns = ["Month", "Avg ER"]

            if len(monthly_ps) >= 2:
                st.markdown('<div class="section-title">📆 Monthly Trend</div>',
                            unsafe_allow_html=True)

                fig2 = make_subplots(rows=1, cols=2,
                                     subplot_titles=["Performance Score Trend",
                                                     "Engagement Rate Trend"])

                fig2.add_trace(go.Scatter(
                    x=monthly_ps["Month"], y=monthly_ps["Avg Score"],
                    mode="lines+markers", line=dict(color="#60a5fa", width=2.5),
                    fill="tozeroy", fillcolor="rgba(96,165,250,0.08)"
                ), row=1, col=1)

                fig2.add_trace(go.Scatter(
                    x=monthly_er["Month"], y=monthly_er["Avg ER"],
                    mode="lines+markers", line=dict(color="#a78bfa", width=2.5),
                    fill="tozeroy", fillcolor="rgba(167,139,250,0.08)"
                ), row=1, col=2)

                fig2.add_hline(y=benchmark_er, line_dash="dot", line_color="#fbbf24",
                               annotation_text=f"Benchmark {benchmark_er}%", row=1, col=2)

                fig2.update_layout(
                    showlegend=False,
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font_color="rgba(255,255,255,0.7)",
                    margin=dict(t=40,b=20,l=10,r=10)
                )
                st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 4 — PREDICTOR
# ══════════════════════════════════════════════════════════════════════════

with tab_predictor:
    st.markdown("""
    <div style="font-size:26px;font-weight:700;color:white;margin-bottom:6px">Engagement Predictor</div>
    <div style="color:rgba(255,255,255,0.45);font-size:14px;margin-bottom:16px">
        Predicts both Performance Score and Engagement Rate before you publish
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="alert alert-purple">
        💡 <strong>How to interpret predictions:</strong><br>
        🔥 Score high + ER high → Strong post — publish and boost!<br>
        📢 Score high + ER low → Good reach but lower relative engagement — add a strong CTA<br>
        🎯 Score low + ER high → Good engagement quality but limited reach — consider boosting<br>
        ⚠️ Both low → Review content before publishing
    </div>""", unsafe_allow_html=True)

    @st.cache_resource
    def train_models(_df, _meta):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import GradientBoostingRegressor

        er_col_  = _meta["engagement_rate_col"]
        ps_col_  = _meta["performance_score_col"]
        type_col_= _meta.get("type_col")
        ts_col_  = _meta.get("time_slot_col")

        # Only include features with enough variation
        # Excluded: saves (82% zeros), comments (93% zeros),
        #           shares (74% zeros), follows (99% zeros)
        # Defaults: reach=median(31), views=median(84), likes=mode(4)
        INCLUDE_COLS = {
            _meta.get("reach_col"): 31,   # median — mode only 6.5% of posts
            _meta.get("views_col"): 84,   # median — mode only 4.1% of posts
            _meta.get("likes_col"):  4,   # mode   — 24.4% representative
        }
        num_f = [col for col in INCLUDE_COLS if col and col in _df.columns]
        cat_f = []
        if type_col_ and type_col_ in _df.columns: cat_f.append(type_col_)
        if "weekday" in _df.columns:               cat_f.append("weekday")
        if ts_col_ and ts_col_ in _df.columns:     cat_f.append(ts_col_)

        all_f = num_f + cat_f
        mdf   = _df[all_f + [ps_col_, er_col_]].dropna()
        if len(mdf) < 20: return None, None, None, None

        transformers = [("num", StandardScaler(), num_f)]
        if cat_f:
            transformers.append(
                ("cat", OneHotEncoder(drop="first", handle_unknown="ignore",
                                      sparse_output=False), cat_f))
        pre = ColumnTransformer(transformers, remainder="drop")

        params = dict(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42)

        pipe_ps = Pipeline([("pre", pre), ("model", GradientBoostingRegressor(**params))])
        pipe_ps.fit(mdf[all_f], mdf[ps_col_])

        pipe_er = Pipeline([("pre", pre), ("model", GradientBoostingRegressor(**params))])
        pipe_er.fit(mdf[all_f], mdf[er_col_])

        return pipe_ps, pipe_er, num_f, cat_f

    pipe_ps, pipe_er, num_feats, cat_feats = train_models(df, meta)

    if pipe_ps is None:
        st.warning("Not enough data to train the model. Need at least 20 posts.")
    else:
        form_col, result_col = st.columns([1, 1])

        with form_col:
            st.markdown('<div class="section-title">Post Details</div>', unsafe_allow_html=True)
            new_post = {}

            # Default values based on historical data analysis
            DEFAULTS = {
                meta.get("reach_col"): (31, "median — most representative"),
                meta.get("views_col"): (84, "median — most representative"),
                meta.get("likes_col"): ( 4, "mode   — 24.4% of posts"),
            }
            for col_name in num_feats:
                label      = col_name.replace("_"," ").title()
                default_v, default_note = DEFAULTS.get(col_name, (0, ""))
                new_post[col_name] = st.number_input(
                    f"Expected {label}",
                    min_value=0, value=default_v, step=1,
                    help=f"Default: {default_v} ({default_note})"
                )

            if type_col and type_col in cat_feats:
                opts = sorted(df[type_col].dropna().unique().tolist())
                new_post[type_col] = st.selectbox("Content Type", opts)

            if "weekday" in cat_feats:
                new_post["weekday"] = st.selectbox(
                    "Posting Day", WEEKDAY_ORDER,
                    index=WEEKDAY_ORDER.index(best_day) if best_day in WEEKDAY_ORDER else 0
                )

            if time_slot_col and time_slot_col in cat_feats:
                slot_options = [s for s in [
                    "🌅 Early Morning (6-9)",
                    "☀️ Late Morning (10-12)",
                    "🌤️ Afternoon (13-17)",
                    "🌆 Evening (18-21)",
                    "🌙 Night (22-5)"
                ] if s in df[time_slot_col].cat.categories]
                new_post[time_slot_col] = st.selectbox("Time Slot", slot_options)

            predict_btn = st.button("✨ Predict Performance",
                                    type="primary", use_container_width=True)

        with result_col:
            st.markdown('<div class="section-title">Prediction</div>', unsafe_allow_html=True)

            if predict_btn:
                pred_ps = max(0, pipe_ps.predict(pd.DataFrame([new_post]))[0])
                pred_er = max(0, pipe_er.predict(pd.DataFrame([new_post]))[0])

                # Determine scenario
                if pred_ps >= avg_ps and pred_er >= avg_er:
                    icon, label, color = "🔥", "STRONG POST", "#4ade80"
                    tip = "Publish and consider boosting to maximize reach!"
                elif pred_ps >= avg_ps and pred_er < avg_er:
                    icon, label, color = "📢", "GOOD REACH", "#60a5fa"
                    tip = "Good absolute impact — add a strong CTA to drive more interactions."
                elif pred_ps < avg_ps and pred_er >= avg_er:
                    icon, label, color = "🎯", "GOOD ENGAGEMENT", "#a78bfa"
                    tip = "Good engagement quality — consider boosting to amplify reach."
                else:
                    icon, label, color = "⚠️", "BELOW AVERAGE", "#fbbf24"
                    tip = "Review content type, timing, or caption before publishing."

                st.markdown(f"""
                <div class="predict-box">
                    <div style="font-size:32px">{icon}</div>
                    <div class="predict-label">PERFORMANCE SCORE</div>
                    <div class="predict-number">{pred_ps:.1f}<span style="font-size:20px">/100</span></div>
                    <div style="margin-top:12px;color:rgba(255,255,255,0.6);font-size:13px">
                        ENGAGEMENT RATE: <strong style="color:white">{pred_er:.1f}%</strong>
                    </div>
                    <div style="margin-top:8px">
                        <span style="background:rgba(255,255,255,0.1);color:{color};
                              border:1px solid {color}40;border-radius:999px;
                              padding:4px 14px;font-size:12px;font-weight:700">
                            {label}
                        </span>
                    </div>
                </div>""", unsafe_allow_html=True)

                st.markdown("")
                m1, m2, m3 = st.columns(3)
                m1.metric("vs Your Score Avg", f"{avg_ps:.1f}",
                           f"{pred_ps-avg_ps:+.1f}")
                m2.metric("vs ER Benchmark",   f"{benchmark_er:.1f}%",
                           f"{pred_er-benchmark_er:+.1f}%")
                m3.metric("Your Best Score",    f"{ps.max():.1f}")

                st.markdown(f"""
                <div class="alert alert-{'green' if pred_ps >= avg_ps else 'yellow'}">
                    💡 {tip}
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="text-align:center;padding:60px 20px;color:rgba(255,255,255,0.3)">
                    <div style="font-size:48px">🎯</div>
                    <div style="margin-top:12px;font-size:15px">
                        Fill in the details on the left<br>and click Predict
                    </div>
                </div>""", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;color:rgba(255,255,255,0.2);padding:24px 0 8px;font-size:12px">
    Travel Mex Tours · Social Media Analytics v3 · Performance Score primary metric ·
    Built with Python & Streamlit
</div>""", unsafe_allow_html=True)
