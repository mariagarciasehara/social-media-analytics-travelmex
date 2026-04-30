# app.py — Travel Mex Tours | Social Media Analytics Dashboard v2
import sys, warnings, io
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from load_data import load_instagram_csv
from preprocessing import preprocess_df

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Travel Mex | Analytics",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BENCHMARK    = 5.0
WEEKDAY_ORDER = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

# ── CSS — works in BOTH light and dark mode ────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    /* ── KPI cards ── */
    .kpi-card {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 14px;
        padding: 20px 22px;
        margin-bottom: 4px;
    }
    .kpi-label {
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: rgba(255,255,255,0.55);
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 32px;
        font-weight: 700;
        color: #ffffff;
        line-height: 1;
    }
    .kpi-delta {
        font-size: 12px;
        margin-top: 6px;
        font-weight: 500;
    }
    .kpi-delta.up   { color: #4ade80; }
    .kpi-delta.down { color: #f87171; }
    .kpi-delta.info { color: #93c5fd; }

    /* ── Section titles ── */
    .section-title {
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: rgba(255,255,255,0.4);
        margin: 28px 0 14px 2px;
    }

    /* ── Insight pills ── */
    .pill {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 600;
        margin: 4px 4px 4px 0;
    }
    .pill-green  { background: rgba(74,222,128,0.15); color: #4ade80; border: 1px solid rgba(74,222,128,0.3); }
    .pill-blue   { background: rgba(96,165,250,0.15); color: #60a5fa; border: 1px solid rgba(96,165,250,0.3); }
    .pill-yellow { background: rgba(251,191,36,0.15); color: #fbbf24; border: 1px solid rgba(251,191,36,0.3); }
    .pill-pink   { background: rgba(244,114,182,0.15); color: #f472b6; border: 1px solid rgba(244,114,182,0.3); }

    /* ── Alert banner ── */
    .alert {
        padding: 14px 18px;
        border-radius: 10px;
        font-size: 14px;
        font-weight: 500;
        margin: 12px 0;
        line-height: 1.5;
    }
    .alert-green  { background: rgba(74,222,128,0.12); border: 1px solid rgba(74,222,128,0.25); color: #86efac; }
    .alert-yellow { background: rgba(251,191,36,0.12); border: 1px solid rgba(251,191,36,0.25); color: #fde68a; }
    .alert-blue   { background: rgba(96,165,250,0.12); border: 1px solid rgba(96,165,250,0.25); color: #bfdbfe; }

    /* ── Predict box ── */
    .predict-box {
        background: linear-gradient(135deg, rgba(24,119,242,0.25) 0%, rgba(131,58,180,0.25) 100%);
        border: 1px solid rgba(131,58,180,0.4);
        border-radius: 16px;
        padding: 28px;
        text-align: center;
    }
    .predict-number {
        font-size: 56px;
        font-weight: 700;
        color: #ffffff;
        line-height: 1;
    }
    .predict-label {
        font-size: 13px;
        color: rgba(255,255,255,0.6);
        margin-top: 6px;
        letter-spacing: 0.05em;
    }
    .predict-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 700;
        margin-top: 10px;
        letter-spacing: 0.05em;
    }

    /* ── Top posts table ── */
    .top-post-row {
        display: flex;
        align-items: center;
        padding: 10px 14px;
        border-radius: 8px;
        margin: 4px 0;
        background: rgba(255,255,255,0.05);
        font-size: 14px;
    }
    .top-post-rank {
        font-size: 18px;
        font-weight: 700;
        color: rgba(255,255,255,0.3);
        width: 36px;
    }
    .top-post-er {
        font-size: 18px;
        font-weight: 700;
        color: #4ade80;
        margin-left: auto;
    }

    /* ── Hide Streamlit chrome ── */
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


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ✈️ Travel Mex Tours")
    st.markdown("---")

    uploaded = st.file_uploader(
        "Upload Instagram CSV",
        type=["csv"],
        help="Export from Instagram Insights or Meta Business Suite"
    )

    st.markdown("---")
    benchmark = st.slider(
        "Benchmark ER (%)",
        min_value=1.0, max_value=15.0, value=BENCHMARK, step=0.5,
        help="Industry avg for 1k–10k follower accounts: 3–6%"
    )
    st.caption(f"👥 Followers: 2,608  •  Category: Travel")
    st.caption("📊 Platform: Instagram")

# ── Load data ─────────────────────────────────────────────────────────────
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
        <h2 style="color:white;margin:16px 0 8px">Travel Mex Analytics</h2>
        <p style="color:rgba(255,255,255,0.5);font-size:16px">
            Upload your Instagram CSV from the sidebar to get started
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Shortcuts ─────────────────────────────────────────────────────────────
er_col       = meta["engagement_rate_col"]
date_col     = meta.get("date_col")
type_col     = meta.get("type_col")
likes_col    = meta.get("likes_col")
saves_col    = meta.get("saves_col")
comments_col = meta.get("comments_col")
shares_col   = meta.get("shares_col")
er           = df[er_col].dropna()
avg_er       = round(er.mean(), 2)
best_day     = df.groupby("weekday")[er_col].mean().idxmax() if "weekday" in df.columns else "N/A"
best_hour    = int(df.groupby("hour")[er_col].mean().idxmax()) if "hour" in df.columns else None
best_type    = df.groupby(type_col)[er_col].mean().idxmax() if type_col and type_col in df.columns else "N/A"


# ══════════════════════════════════════════════════════════════════════════
# NAVIGATION TABS
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

    # ── Page header ───────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="margin-bottom:24px">
        <div style="font-size:26px;font-weight:700;color:white">Performance Overview</div>
        <div style="color:rgba(255,255,255,0.45);font-size:14px;margin-top:4px">
            {len(df)} posts  ·  {meta.get("date_range_start","?")} → {meta.get("date_range_end","?")}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI row ────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)

    delta_bm = round(avg_er - benchmark, 1)
    delta_cls = "up" if delta_bm >= 0 else "down"
    delta_str = f"{'▲' if delta_bm>=0 else '▼'} {abs(delta_bm):.1f}% vs {benchmark}% benchmark"

    with k1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Avg Engagement Rate</div>
            <div class="kpi-value">{avg_er:.1f}%</div>
            <div class="kpi-delta {delta_cls}">{delta_str}</div>
        </div>""", unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Posts Analyzed</div>
            <div class="kpi-value">{len(df)}</div>
            <div class="kpi-delta info">📅 {meta.get("date_range_start","?")} – {meta.get("date_range_end","?")}</div>
        </div>""", unsafe_allow_html=True)

    with k3:
        best_day_er = round(df.groupby("weekday")[er_col].mean().get(best_day, 0), 1) if "weekday" in df.columns else 0
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Best Day to Post</div>
            <div class="kpi-value" style="font-size:24px">{best_day}</div>
            <div class="kpi-delta up">▲ {best_day_er}% avg ER</div>
        </div>""", unsafe_allow_html=True)

    with k4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Best Content Type</div>
            <div class="kpi-value" style="font-size:22px">{best_type}</div>
            <div class="kpi-delta up">▲ Highest avg ER</div>
        </div>""", unsafe_allow_html=True)

    with k5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Best Hour to Post</div>
            <div class="kpi-value">{f"{best_hour}:00" if best_hour is not None else "N/A"}</div>
            <div class="kpi-delta info">Peak engagement hour</div>
        </div>""", unsafe_allow_html=True)

    # ── Benchmark alert ────────────────────────────────────────────────────
    st.markdown("")
    if avg_er >= benchmark * 1.5:
        mult = round(avg_er / benchmark, 1)
        st.markdown(f"""
        <div class="alert alert-green">
            ✅ <strong>Outstanding performance!</strong>
            Your {avg_er:.1f}% engagement rate is <strong>{mult}× above</strong> the {benchmark}% benchmark
            for accounts with 1k–10k followers. Your audience is highly engaged.
        </div>""", unsafe_allow_html=True)
    elif avg_er >= benchmark:
        st.markdown(f"""
        <div class="alert alert-green">
            ✅ Your {avg_er:.1f}% engagement rate is above the {benchmark}% benchmark. Good work!
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="alert alert-yellow">
            ⚠️ Your {avg_er:.1f}% engagement rate is below the {benchmark}% benchmark.
            Check the recommendations in each tab.
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # ── Charts row ─────────────────────────────────────────────────────────
    chart_col1, chart_col2 = st.columns([3, 2])

    with chart_col1:
        # ER distribution
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=er, nbinsx=25,
            marker_color="rgba(96,165,250,0.7)",
            marker_line_color="rgba(96,165,250,1)",
            marker_line_width=1,
            name="Posts"
        ))
        fig.add_vline(x=avg_er, line_dash="dash", line_color="#4ade80",
                      annotation_text=f"Your avg {avg_er:.1f}%",
                      annotation_font_color="#4ade80")
        fig.add_vline(x=benchmark, line_dash="dot", line_color="#fbbf24",
                      annotation_text=f"Benchmark {benchmark}%",
                      annotation_font_color="#fbbf24")
        fig.update_layout(
            title="Engagement Rate Distribution",
            xaxis_title="Engagement Rate (%)",
            yaxis_title="Number of Posts",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="rgba(255,255,255,0.7)",
            showlegend=False,
            margin=dict(t=40, b=20, l=10, r=10)
        )
        st.plotly_chart(fig, use_container_width=True)

    with chart_col2:
        # Engagement breakdown pie
        breakdown = {k: v for k, v in {
            "Likes": likes_col, "Comments": comments_col,
            "Saves": saves_col, "Shares": shares_col,
        }.items() if v and v in df.columns}

        if breakdown:
            totals = {k: int(df[v].sum()) for k, v in breakdown.items()}
            fig2 = go.Figure(go.Pie(
                labels=list(totals.keys()),
                values=list(totals.values()),
                hole=0.5,
                marker_colors=["#60a5fa","#a78bfa","#f472b6","#fbbf24"],
                textinfo="percent",
                textfont_size=13
            ))
            fig2.update_layout(
                title="Interaction Breakdown",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="rgba(255,255,255,0.7)",
                legend=dict(font=dict(color="rgba(255,255,255,0.6)")),
                margin=dict(t=40, b=0, l=0, r=0)
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── Top 5 posts ────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🏆 Top 5 Posts by Engagement Rate</div>', unsafe_allow_html=True)

    top5 = df.nlargest(5, er_col)
    medals = ["🥇","🥈","🥉","4️⃣","5️⃣"]

    for i, (_, row) in enumerate(top5.iterrows()):
        date_str = str(row[date_col].date()) if date_col and pd.notna(row.get(date_col)) else "N/A"
        type_str = str(row[type_col]) if type_col and pd.notna(row.get(type_col)) else ""
        eng_str  = f"{int(row['total_engagements'])} interactions" if pd.notna(row.get('total_engagements')) else ""

        st.markdown(f"""
        <div class="top-post-row">
            <div class="top-post-rank">{medals[i]}</div>
            <div style="color:rgba(255,255,255,0.85)">
                <span style="font-weight:600">{date_str}</span>
                {"&nbsp;·&nbsp;" + type_str if type_str else ""}
                {"&nbsp;·&nbsp;" + eng_str if eng_str else ""}
            </div>
            <div class="top-post-er">{row[er_col]:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — BEST TIME
# ══════════════════════════════════════════════════════════════════════════

with tab_timing:
    st.markdown("""
    <div style="font-size:26px;font-weight:700;color:white;margin-bottom:6px">Best Time to Post</div>
    <div style="color:rgba(255,255,255,0.45);font-size:14px;margin-bottom:24px">
        Based on historical engagement data from your account
    </div>
    """, unsafe_allow_html=True)

    if "weekday" not in df.columns:
        st.info("No date data found in your CSV.")
    else:
        day_avg = (
            df.groupby("weekday")[er_col].mean()
              .reindex([d for d in WEEKDAY_ORDER if d in df["weekday"].unique()])
              .reset_index()
        )
        day_avg.columns = ["Day","Avg ER"]

        col_l, col_r = st.columns(2)

        with col_l:
            colors = ["#4ade80" if d == best_day else "#3b82f6" for d in day_avg["Day"]]
            fig = go.Figure(go.Bar(
                x=day_avg["Day"], y=day_avg["Avg ER"],
                marker_color=colors,
                text=day_avg["Avg ER"].round(1).astype(str) + "%",
                textposition="outside",
                textfont=dict(color="white", size=11)
            ))
            fig.add_hline(y=benchmark, line_dash="dot", line_color="#fbbf24",
                          annotation_text=f"Benchmark {benchmark}%",
                          annotation_font_color="#fbbf24")
            fig.update_layout(
                title=f"Best day: <b>{best_day}</b> 🏆",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="rgba(255,255,255,0.7)",
                yaxis=dict(gridcolor="rgba(255,255,255,0.07)"),
                margin=dict(t=50,b=20,l=10,r=10)
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            if "hour" in df.columns:
                hour_avg = df.groupby("hour")[er_col].mean().reset_index()
                hour_avg.columns = ["Hour","Avg ER"]
                top3_hours = hour_avg.nlargest(3,"Avg ER")["Hour"].tolist()

                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=hour_avg["Hour"], y=hour_avg["Avg ER"],
                    mode="lines+markers",
                    line=dict(color="#60a5fa", width=2.5),
                    marker=dict(size=7, color="#60a5fa"),
                    fill="tozeroy",
                    fillcolor="rgba(96,165,250,0.08)"
                ))
                for h in top3_hours:
                    fig2.add_vline(x=h, line_dash="dot", line_color="#f472b6",
                                   annotation_text=f"{int(h)}:00 ⭐",
                                   annotation_font_color="#f472b6")
                fig2.update_layout(
                    title=f"Best hours: {', '.join([str(int(h))+':00' for h in top3_hours])}",
                    xaxis_title="Hour of Day",
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font_color="rgba(255,255,255,0.7)",
                    yaxis=dict(gridcolor="rgba(255,255,255,0.07)"),
                    margin=dict(t=50,b=20,l=10,r=10)
                )
                st.plotly_chart(fig2, use_container_width=True)

        # Recommendations
        top3_str = ", ".join([f"{int(h)}:00" for h in top3_hours]) if "hour" in df.columns else "N/A"
        st.markdown(f"""
        <div class="alert alert-blue">
            📅 <strong>Recommendation:</strong>
            Post on <strong>{best_day}</strong> at <strong>{top3_str}</strong>
            to maximize your engagement rate.
        </div>""", unsafe_allow_html=True)

        # Weekend vs Weekday
        if "is_weekend" in df.columns:
            st.markdown('<div class="section-title">Weekend vs Weekday</div>', unsafe_allow_html=True)
            wk_df = df.copy()
            wk_df["Period"] = wk_df["is_weekend"].map({True: "Weekend 🌅", False: "Weekday 💼"})
            wk_avg = wk_df.groupby("Period")[er_col].mean().reset_index()

            fig3 = go.Figure(go.Bar(
                x=wk_avg["Period"], y=wk_avg[er_col].round(2),
                marker_color=["#a78bfa","#60a5fa"],
                text=wk_avg[er_col].round(1).astype(str) + "%",
                textposition="outside",
                textfont=dict(color="white", size=13)
            ))
            fig3.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="rgba(255,255,255,0.7)",
                yaxis=dict(gridcolor="rgba(255,255,255,0.07)"),
                margin=dict(t=20,b=20,l=10,r=10),
                showlegend=False
            )
            st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — CONTENT PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════

with tab_content:
    st.markdown("""
    <div style="font-size:26px;font-weight:700;color:white;margin-bottom:6px">Content Performance</div>
    <div style="color:rgba(255,255,255,0.45);font-size:14px;margin-bottom:24px">
        Which type of content drives the most engagement
    </div>
    """, unsafe_allow_html=True)

    if not type_col or type_col not in df.columns:
        st.info("No content type column found in your CSV.")
    else:
        type_stats = (
            df.groupby(type_col)[er_col]
              .agg(["mean","median","count","std"])
              .rename(columns={"mean":"Avg ER","median":"Median ER","count":"Posts","std":"Std Dev"})
              .sort_values("Avg ER", ascending=False)
              .reset_index()
              .round(2)
        )

        col_l, col_r = st.columns([3,2])

        with col_l:
            colors = ["#4ade80" if t == best_type else "#3b82f6" for t in type_stats[type_col]]
            fig = go.Figure(go.Bar(
                x=type_stats[type_col], y=type_stats["Avg ER"],
                marker_color=colors,
                text=type_stats["Avg ER"].astype(str) + "%",
                textposition="outside",
                textfont=dict(color="white", size=11)
            ))
            fig.add_hline(y=avg_er, line_dash="dash", line_color="rgba(255,255,255,0.3)",
                          annotation_text=f"Your avg {avg_er:.1f}%",
                          annotation_font_color="rgba(255,255,255,0.5)")
            fig.update_layout(
                title=f"Best type: <b>{best_type}</b> 🏆",
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="rgba(255,255,255,0.7)",
                yaxis=dict(gridcolor="rgba(255,255,255,0.07)"),
                margin=dict(t=50,b=20,l=10,r=10)
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            st.markdown('<div class="section-title">Breakdown by Type</div>', unsafe_allow_html=True)
            for _, row in type_stats.iterrows():
                is_best = row[type_col] == best_type
                st.markdown(f"""
                <div style="background:rgba(255,255,255,{'0.08' if is_best else '0.04'});
                     border:1px solid rgba(255,255,255,{'0.2' if is_best else '0.08'});
                     border-radius:10px; padding:12px 16px; margin-bottom:8px;">
                    <div style="display:flex;justify-content:space-between;align-items:center">
                        <span style="font-weight:600;color:white">
                            {'⭐ ' if is_best else ''}{row[type_col]}
                        </span>
                        <span style="font-size:20px;font-weight:700;color:#4ade80">{row['Avg ER']}%</span>
                    </div>
                    <div style="color:rgba(255,255,255,0.4);font-size:12px;margin-top:4px">
                        {int(row['Posts'])} posts · Median {row['Median ER']}% · Std ±{row['Std Dev']}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Monthly trend
        if date_col and pd.api.types.is_datetime64_any_dtype(df[date_col]):
            monthly = df.set_index(date_col)[er_col].resample("ME").mean().dropna().reset_index()
            monthly.columns = ["Month","Avg ER"]

            if len(monthly) >= 2:
                st.markdown('<div class="section-title">📆 Monthly Trend</div>', unsafe_allow_html=True)
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=monthly["Month"], y=monthly["Avg ER"],
                    mode="lines+markers",
                    line=dict(color="#60a5fa", width=2.5),
                    marker=dict(size=8, color="#60a5fa"),
                    fill="tozeroy",
                    fillcolor="rgba(96,165,250,0.08)"
                ))
                fig2.add_hline(y=benchmark, line_dash="dot", line_color="#fbbf24",
                               annotation_text=f"Benchmark {benchmark}%",
                               annotation_font_color="#fbbf24")
                fig2.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font_color="rgba(255,255,255,0.7)",
                    yaxis=dict(gridcolor="rgba(255,255,255,0.07)"),
                    margin=dict(t=20,b=20,l=10,r=10)
                )
                st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 4 — ENGAGEMENT PREDICTOR
# ══════════════════════════════════════════════════════════════════════════

with tab_predictor:
    st.markdown("""
    <div style="font-size:26px;font-weight:700;color:white;margin-bottom:6px">Engagement Predictor</div>
    <div style="color:rgba(255,255,255,0.45);font-size:14px;margin-bottom:24px">
        Fill in your post details and our AI will predict its Engagement Rate before you publish
    </div>
    """, unsafe_allow_html=True)

    # Train model
    @st.cache_resource
    def train_model(_df, _meta):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import GradientBoostingRegressor

        numeric_keys = ["reach_col","views_col","likes_col","comments_col",
                        "saves_col","shares_col","follows_col"]
        num_feats = [_meta[k] for k in numeric_keys if _meta.get(k) and _meta[k] in _df.columns]
        cat_feats = []
        if _meta.get("type_col") and _meta["type_col"] in _df.columns:
            cat_feats.append(_meta["type_col"])
        if "weekday" in _df.columns: cat_feats.append("weekday")
        if "hour"    in _df.columns: cat_feats.append("hour")

        all_feats = num_feats + cat_feats
        mdf = _df[all_feats + [_meta["engagement_rate_col"]]].dropna()
        if len(mdf) < 20:
            return None, None, None

        transformers = [("num", StandardScaler(), num_feats)]
        if cat_feats:
            transformers.append(("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), cat_feats))

        pipe = Pipeline([
            ("pre", ColumnTransformer(transformers, remainder="drop")),
            ("model", GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42))
        ])
        pipe.fit(mdf[all_feats], mdf[_meta["engagement_rate_col"]])
        return pipe, num_feats, cat_feats

    pipe, num_feats, cat_feats = train_model(df, meta)

    if pipe is None:
        st.warning("Not enough data to train the model. Need at least 20 posts.")
    else:
        form_col, result_col = st.columns([1, 1])

        with form_col:
            st.markdown('<div class="section-title">Post Details</div>', unsafe_allow_html=True)
            new_post = {}

            for col_name in num_feats:
                label   = col_name.replace("_"," ").title()
                med_val = int(df[col_name].median()) if col_name in df.columns else 0
                new_post[col_name] = st.number_input(
                    f"Expected {label}",
                    min_value=0, value=med_val, step=1,
                    help=f"Your historical median: {med_val}"
                )

            if type_col and type_col in cat_feats:
                opts = sorted(df[type_col].dropna().unique().tolist())
                new_post[type_col] = st.selectbox("Content Type", opts)

            if "weekday" in cat_feats:
                new_post["weekday"] = st.selectbox(
                    "Posting Day",
                    WEEKDAY_ORDER,
                    index=WEEKDAY_ORDER.index(best_day) if best_day in WEEKDAY_ORDER else 0
                )

            if "hour" in cat_feats:
                new_post["hour"] = st.slider("Posting Hour", 0, 23,
                                              value=best_hour if best_hour else 9)

            predict_btn = st.button("✨ Predict Engagement Rate",
                                    type="primary", use_container_width=True)

        with result_col:
            st.markdown('<div class="section-title">Prediction</div>', unsafe_allow_html=True)

            if predict_btn:
                prediction = max(0, pipe.predict(pd.DataFrame([new_post]))[0])

                if prediction >= avg_er * 1.1:
                    badge_color = "#4ade80"
                    badge_bg    = "rgba(74,222,128,0.2)"
                    badge_text  = "🔥 ABOVE YOUR AVERAGE"
                elif prediction >= benchmark:
                    badge_color = "#60a5fa"
                    badge_bg    = "rgba(96,165,250,0.2)"
                    badge_text  = "✅ ABOVE BENCHMARK"
                else:
                    badge_color = "#fbbf24"
                    badge_bg    = "rgba(251,191,36,0.2)"
                    badge_text  = "⚠️ BELOW BENCHMARK"

                st.markdown(f"""
                <div class="predict-box">
                    <div class="predict-label">PREDICTED ENGAGEMENT RATE</div>
                    <div class="predict-number">{prediction:.1f}%</div>
                    <div>
                        <span class="predict-badge"
                              style="background:{badge_bg};color:{badge_color};border:1px solid {badge_color}40">
                            {badge_text}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("")
                m1, m2, m3 = st.columns(3)
                delta1 = round(prediction - avg_er, 1)
                delta2 = round(prediction - benchmark, 1)
                m1.metric("vs Your Avg",   f"{avg_er:.1f}%",   f"{delta1:+.1f}%")
                m2.metric("vs Benchmark",  f"{benchmark:.1f}%",f"{delta2:+.1f}%")
                m3.metric("Your Best Post", f"{er.max():.1f}%")

                st.markdown("")
                if prediction < avg_er:
                    st.markdown("""
                    <div class="alert alert-yellow">
                        💡 <strong>Tip:</strong> Try posting on your best day/hour
                        or switching content type to improve predicted performance.
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="alert alert-green">
                        💡 <strong>Tip:</strong> This looks like a strong post!
                        Consider boosting it to maximize reach.
                    </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="text-align:center;padding:60px 20px;color:rgba(255,255,255,0.3)">
                    <div style="font-size:48px">🎯</div>
                    <div style="margin-top:12px;font-size:15px">
                        Fill in the details on the left<br>and click Predict
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;color:rgba(255,255,255,0.2);padding:24px 0 8px;font-size:12px">
    Travel Mex Tours · Social Media Analytics · Built with Python & Streamlit · Data is private and not stored
</div>
""", unsafe_allow_html=True)
