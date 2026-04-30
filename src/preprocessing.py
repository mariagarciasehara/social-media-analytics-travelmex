import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import warnings


def preprocess_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Advanced and robust preprocessing for Instagram Insights data.
    - Detects publish_time before date (fixes empty date column issue)
    - Generates full metadata including date_range, avg_engagement_rate, etc.
    - Adds temporal features: weekday, hour, month, is_weekend
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

    # Fallback: partial match
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
    # 2. Smart and flexible key column search
    # -------------------------------------------------------------------
    def find_column(keywords: list) -> Optional[str]:
        """Find column by exact or partial match (case-insensitive)"""
        lowered = [c.lower() for c in df.columns]
        for kw in keywords:
            if kw.lower() in lowered:
                idx = lowered.index(kw.lower())
                return df.columns[idx]
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

    # -------------------------------------------------------------------
    # 3. Convert metric columns to numeric
    # -------------------------------------------------------------------
    metric_cols = [
        reach_col, impressions_col, views_col, likes_col, comments_col,
        saves_col, shares_col, follows_col, profile_visits
    ]

    for col in metric_cols:
        if col is not None and col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(r"[,%$]", "", regex=True)
                .str.strip()
                .replace({"nan": np.nan, "": np.nan, "<NA>": np.nan})
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # -------------------------------------------------------------------
    # 4. Total engagements
    # -------------------------------------------------------------------
    engagement_components = [c for c in [likes_col, comments_col, saves_col,
                                          shares_col, follows_col] if c]
    if engagement_components:
        df["total_engagements"] = df[engagement_components].clip(lower=0).sum(axis=1, skipna=True)
    else:
        df["total_engagements"] = np.nan

    # -------------------------------------------------------------------
    # 5. Engagement Rate (reach > impressions > views)
    # -------------------------------------------------------------------
    denominator      = None
    denominator_name = "unknown"

    for candidate, name in [
        (reach_col,       "reach"),
        (impressions_col, "impressions"),
        (views_col,       "views"),
    ]:
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
    # 6. Advanced KPIs
    # -------------------------------------------------------------------
    if saves_col and denominator:
        df["save_rate_pct"] = (100 * df[saves_col] / df[denominator].replace(0, np.nan)).round(4)

    if shares_col and denominator:
        df["share_rate_pct"] = (100 * df[shares_col] / df[denominator].replace(0, np.nan)).round(4)

    if likes_col and comments_col:
        df["comments_per_like"] = (df[comments_col] / df[likes_col].replace(0, np.nan)).round(4)

    if views_col and likes_col:
        df["like_rate_on_views_pct"] = (
            100 * df[likes_col] / df[views_col].replace(0, np.nan)
        ).round(4)

    if follows_col and profile_visits:
        df["profile_to_follow_conversion_pct"] = (
            100 * df[follows_col] / df[profile_visits].replace(0, np.nan)
        ).round(4)

    # -------------------------------------------------------------------
    # 7. Temporal features
    # -------------------------------------------------------------------
    if date_col and pd.api.types.is_datetime64_any_dtype(df[date_col]):
        valid_dates = df[date_col].notna()
        if valid_dates.sum() > 0:
            df["weekday"]    = df[date_col].dt.day_name()
            df["hour"]       = df[date_col].dt.hour
            df["month"]      = df[date_col].dt.month_name()
            df["is_weekend"] = df[date_col].dt.dayofweek >= 5

    # -------------------------------------------------------------------
    # 8. Column order
    # -------------------------------------------------------------------
    priority = [
        date_col, "weekday", "hour", "month",
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
    # 9. Metadata — includes date_range, avg ER, etc.
    # -------------------------------------------------------------------
    er = df["engagement_rate_pct"].dropna()

    meta = {
        # Column names
        "date_col":            date_col,
        "reach_col":           reach_col,
        "impressions_col":     impressions_col,
        "views_col":           views_col,
        "likes_col":           likes_col,
        "comments_col":        comments_col,
        "saves_col":           saves_col,
        "shares_col":          shares_col,
        "follows_col":         follows_col,
        "profile_visits_col":  profile_visits,
        "type_col":            type_col,
        "duration_col":        duration_col,
        "engagement_col":      "total_engagements",
        "engagement_rate_col": "engagement_rate_pct",
        "engagement_denominator": denominator_name,
        # Summary stats
        "total_rows":             len(df),
        "avg_engagement_rate":    round(er.mean(), 2) if len(er) else None,
        "median_engagement_rate": round(er.median(), 2) if len(er) else None,
        "max_engagement_rate":    round(er.max(), 2) if len(er) else None,
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
    print(f"   → Date column        : {date_col}")
    print(f"   → ER calculated via  : {denominator_name.upper()}")
    print(f"   → Avg ER             : {meta['avg_engagement_rate']}%")
    print(f"   → Date range         : {meta['date_range_start']} → {meta['date_range_end']}")

    return df, meta
