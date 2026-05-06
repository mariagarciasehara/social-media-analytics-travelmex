import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import warnings


def preprocess_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Advanced and robust preprocessing for Instagram Insights data.

    Key features:
    - Detects publish_time before date (fixes empty date column issue)
    - Adds performance_score (0-100) correcting for reach bias
    - Two performance tiers:
        performance_tier          → vs account maximum potential (fixed thresholds)
        performance_tier_relative → vs other posts in the dataset (percentiles)
    - Adds time_slot (franjas horarias) for statistically valid hour analysis
    - Generates full metadata
    """
    df = df.copy()

    # -------------------------------------------------------------------
    # 1. Date column detection (publish_time takes priority over date)
    # -------------------------------------------------------------------
    date_col = None
    date_keywords = ["publish_time", "posted", "published", "fecha", "hora", "date", "time"]
    for kw in date_keywords:
        for col in df.columns:
            if kw.lower() == col.lower():
                date_col = col
                break
        if date_col:
            break

    if not date_col:
        for kw in date_keywords:
            matches = [c for c in df.columns if kw.lower() in c.lower()]
            if matches:
                date_col = matches[0]
                break

    if date_col:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)

    # -------------------------------------------------------------------
    # 2. Key column detection
    # -------------------------------------------------------------------
    def find_column(keywords: list) -> Optional[str]:
        lowered = [c.lower() for c in df.columns]
        for kw in keywords:
            if kw.lower() in lowered:
                return df.columns[lowered.index(kw.lower())]
            matches = [c for c in df.columns if kw.lower() in c.lower()]
            if matches:
                return matches[0]
        return None

    reach_col       = find_column(["reach", "alcance", "people reached"])
    impressions_col = find_column(["impressions", "impresiones"])
    views_col       = find_column(["views", "reproducciones", "plays", "video_views"])
    likes_col       = find_column(["likes", "me gusta"])
    comments_col    = find_column(["comments", "comentarios"])
    saves_col       = find_column(["saves", "guardados", "bookmarks"])
    shares_col      = find_column(["shares", "compartidos"])
    follows_col     = find_column(["follows", "new follows", "profile_activity"])
    profile_visits  = find_column(["profile_visits", "profile visits", "visitas al perfil"])
    type_col        = find_column(["post_type", "media_type", "type", "tipo"])
    duration_col    = find_column(["duration_sec", "duration", "video_duration", "length"])

    # Clean invalid post type values in the raw data
    if type_col and type_col in df.columns:
        invalid_types = ["Post type", "post type", "Post Type"]
        df[type_col] = df[type_col].replace(invalid_types, np.nan)

    # -------------------------------------------------------------------
    # 3. Numeric conversion
    # -------------------------------------------------------------------
    metric_cols = [reach_col, impressions_col, views_col, likes_col, comments_col,
                   saves_col, shares_col, follows_col, profile_visits]

    for col in metric_cols:
        if col and col in df.columns:
            df[col] = (
                df[col].astype(str)
                       .str.replace(r"[,%$]", "", regex=True)
                       .str.strip()
                       .replace({"nan": np.nan, "": np.nan, "<NA>": np.nan})
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # -------------------------------------------------------------------
    # 4. Total engagements
    # -------------------------------------------------------------------
    eng_components = [c for c in [likes_col, comments_col, saves_col,
                                   shares_col, follows_col] if c]
    if eng_components:
        df["total_engagements"] = df[eng_components].clip(lower=0).sum(axis=1, skipna=True)
    else:
        df["total_engagements"] = np.nan

    # -------------------------------------------------------------------
    # 5. Engagement Rate (reach > impressions > views)
    # -------------------------------------------------------------------
    denominator, denominator_name = None, "unknown"
    for candidate, name in [(reach_col, "reach"), (impressions_col, "impressions"),
                             (views_col, "views")]:
        if candidate and pd.notna(df[candidate]).sum() > len(df) * 0.25:
            denominator, denominator_name = candidate, name
            break

    if denominator:
        df["engagement_rate_pct"] = (
            100 * df["total_engagements"] / df[denominator].replace(0, np.nan)
        ).round(4)
    else:
        df["engagement_rate_pct"] = np.nan

    # -------------------------------------------------------------------
    # 6. Performance Score (0-100) + Two Performance Tiers
    #
    #    Corrects reach bias — a post reaching more people is not
    #    penalized for having lower ER if its absolute impact is high.
    #
    #    Formula:
    #    performance_score = 0.4 * ER_norm
    #                      + 0.4 * engagements_norm
    #                      + 0.2 * reach_norm
    #
    #    Each component normalized to [0,1] using account's own min/max.
    #    Score of 100 = best possible across all three dimensions simultaneously.
    #
    #    Tier 1 — performance_tier (vs account maximum potential):
    #       0-33  → Low    (using up to 33% of account's proven potential)
    #       34-66 → Medium (using 34-66% of account's proven potential)
    #       67-100 → High  (using 67%+ of account's proven potential)
    #
    #    Tier 2 — performance_tier_relative (vs other posts):
    #       Bottom 25% → Below Average
    #       Middle 50% → Average
    #       Top 25%    → Above Average
    # -------------------------------------------------------------------
    def min_max_norm(series: pd.Series) -> pd.Series:
        s_min, s_max = series.min(), series.max()
        if s_max == s_min:
            return pd.Series(np.zeros(len(series)), index=series.index)
        return (series - s_min) / (s_max - s_min)

    er_norm   = min_max_norm(df["engagement_rate_pct"].fillna(0))
    eng_norm  = min_max_norm(df["total_engagements"].fillna(0))
    reach_norm = (
        min_max_norm(df[denominator].fillna(0))
        if denominator and denominator in df.columns
        else pd.Series(np.zeros(len(df)), index=df.index)
    )

    df["performance_score"] = (
        0.4 * er_norm + 0.4 * eng_norm + 0.2 * reach_norm
    ) * 100
    df["performance_score"] = df["performance_score"].round(2)

    # Tier 1 — vs account maximum potential (fixed thresholds)
    df["performance_tier"] = pd.cut(
        df["performance_score"],
        bins=[-1, 33, 66, 101],
        labels=["Low", "Medium", "High"]
    )

    # Tier 2 — vs other posts in the dataset (percentiles)
    p25 = df["performance_score"].quantile(0.25)
    p75 = df["performance_score"].quantile(0.75)
    df["performance_tier_relative"] = pd.cut(
        df["performance_score"],
        bins=[-1, p25, p75, 101],
        labels=["Below Average", "Average", "Above Average"]
    )

    # -------------------------------------------------------------------
    # 7. Advanced KPIs
    # -------------------------------------------------------------------
    if saves_col and denominator:
        df["save_rate_pct"] = (
            100 * df[saves_col] / df[denominator].replace(0, np.nan)
        ).round(4)

    if shares_col and denominator:
        df["share_rate_pct"] = (
            100 * df[shares_col] / df[denominator].replace(0, np.nan)
        ).round(4)

    if likes_col and comments_col:
        df["comments_per_like"] = (
            df[comments_col] / df[likes_col].replace(0, np.nan)
        ).round(4)

    if views_col and likes_col:
        df["like_rate_on_views_pct"] = (
            100 * df[likes_col] / df[views_col].replace(0, np.nan)
        ).round(4)

    if follows_col and profile_visits:
        df["profile_to_follow_conversion_pct"] = (
            100 * df[follows_col] / df[profile_visits].replace(0, np.nan)
        ).round(4)

    # -------------------------------------------------------------------
    # 8. Temporal features + Time Slots (franjas horarias)
    #
    #    Why time slots instead of exact hours?
    #    With 63% of posts at 9:00, comparing individual hours is not
    #    statistically valid. Grouping into slots gives enough samples
    #    per group for meaningful comparisons.
    #
    #    Slots:
    #    Early Morning  →  6:00 -  9:00
    #    Late Morning   → 10:00 - 12:00
    #    Afternoon      → 13:00 - 17:00
    #    Evening        → 18:00 - 21:00
    #    Night          → 22:00 -  5:00
    # -------------------------------------------------------------------
    if date_col and pd.api.types.is_datetime64_any_dtype(df[date_col]):
        if df[date_col].notna().sum() > 0:
            df["weekday"]    = df[date_col].dt.day_name()
            df["hour"]       = df[date_col].dt.hour
            df["month"]      = df[date_col].dt.month_name()
            df["is_weekend"] = df[date_col].dt.dayofweek >= 5

            def assign_time_slot(hour):
                if pd.isna(hour):      return np.nan
                hour = int(hour)
                if   6 <= hour <=  9:  return "🌅 Early Morning (6-9)"
                elif 10 <= hour <= 12: return "☀️ Late Morning (10-12)"
                elif 13 <= hour <= 17: return "🌤️ Afternoon (13-17)"
                elif 18 <= hour <= 21: return "🌆 Evening (18-21)"
                else:                  return "🌙 Night (22-5)"

            df["time_slot"] = df["hour"].apply(assign_time_slot)

            TIME_SLOT_ORDER = [
                "🌅 Early Morning (6-9)",
                "☀️ Late Morning (10-12)",
                "🌤️ Afternoon (13-17)",
                "🌆 Evening (18-21)",
                "🌙 Night (22-5)"
            ]
            df["time_slot"] = pd.Categorical(
                df["time_slot"],
                categories=TIME_SLOT_ORDER,
                ordered=True
            )

    # -------------------------------------------------------------------
    # 9. Column order
    # -------------------------------------------------------------------
    priority = [
        date_col, "weekday", "hour", "time_slot", "month",
        "performance_score", "performance_tier", "performance_tier_relative",
        "engagement_rate_pct", "total_engagements",
        reach_col, impressions_col, views_col,
        likes_col, comments_col, saves_col, shares_col, follows_col,
        "save_rate_pct", "share_rate_pct", "comments_per_like",
        "like_rate_on_views_pct", type_col, duration_col,
    ]
    priority = [c for c in priority if c and c in df.columns]
    rest     = [c for c in df.columns if c not in priority]
    df       = df[priority + rest]

    # -------------------------------------------------------------------
    # 10. Metadata
    # -------------------------------------------------------------------
    er_col_name = "engagement_rate_pct"
    er = df[er_col_name].dropna()
    ps = df["performance_score"].dropna()

    # Time slot summary
    time_slot_summary = {}
    if "time_slot" in df.columns:
        ts = df.groupby("time_slot", observed=True).agg(
            posts=(er_col_name, "count"),
            avg_er=(er_col_name, "mean"),
            avg_score=("performance_score", "mean")
        ).round(2)
        time_slot_summary = ts.to_dict("index")

    meta = {
        # Column names
        "date_col":                    date_col,
        "reach_col":                   reach_col,
        "impressions_col":             impressions_col,
        "views_col":                   views_col,
        "likes_col":                   likes_col,
        "comments_col":                comments_col,
        "saves_col":                   saves_col,
        "shares_col":                  shares_col,
        "follows_col":                 follows_col,
        "profile_visits_col":          profile_visits,
        "type_col":                    type_col,
        "duration_col":                duration_col,
        "engagement_col":              "total_engagements",
        "engagement_rate_col":         er_col_name,
        "performance_score_col":       "performance_score",
        "performance_tier_col":        "performance_tier",
        "performance_tier_relative_col": "performance_tier_relative",
        "time_slot_col":               "time_slot" if "time_slot" in df.columns else None,
        "engagement_denominator":      denominator_name,
        # Summary stats
        "total_rows":                  len(df),
        "avg_engagement_rate":         round(er.mean(), 2) if len(er) else None,
        "median_engagement_rate":      round(er.median(), 2) if len(er) else None,
        "max_engagement_rate":         round(er.max(), 2) if len(er) else None,
        "avg_performance_score":       round(ps.mean(), 2) if len(ps) else None,
        # Tier 1 — vs maximum potential
        "high_performers_count":       int((df["performance_tier"] == "High").sum()),
        "medium_performers_count":     int((df["performance_tier"] == "Medium").sum()),
        "low_performers_count":        int((df["performance_tier"] == "Low").sum()),
        # Tier 2 — vs other posts
        "above_avg_count":             int((df["performance_tier_relative"] == "Above Average").sum()),
        "avg_count":                   int((df["performance_tier_relative"] == "Average").sum()),
        "below_avg_count":             int((df["performance_tier_relative"] == "Below Average").sum()),
        "time_slot_summary":           time_slot_summary,
        "date_range_start": (
            df[date_col].min().date().isoformat()
            if date_col and df[date_col].notna().any() else None
        ),
        "date_range_end": (
            df[date_col].max().date().isoformat()
            if date_col and df[date_col].notna().any() else None
        ),
    }

    print("✅ Preprocessing completed!")
    print(f"   → Date column            : {date_col}")
    print(f"   → ER calculated via      : {denominator_name.upper()}")
    print(f"   → Avg ER                 : {meta['avg_engagement_rate']}%")
    print(f"   → Avg Performance Score  : {meta['avg_performance_score']}/100")
    print(f"   → vs Maximum Potential   : High={meta['high_performers_count']} | Medium={meta['medium_performers_count']} | Low={meta['low_performers_count']}")
    print(f"   → vs Other Posts         : Above Avg={meta['above_avg_count']} | Avg={meta['avg_count']} | Below Avg={meta['below_avg_count']}")
    if time_slot_summary:
        print(f"   → Time slots:")
        for slot, stats in time_slot_summary.items():
            print(f"      {slot}: {stats['posts']} posts | avg ER {stats['avg_er']}%")

    return df, meta
