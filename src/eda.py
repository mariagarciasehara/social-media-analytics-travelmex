# eda.py
import os
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Professional styling
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (11, 6)
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 13
plt.rcParams["xtick.labelsize"] = 11
plt.rcParams["ytick.labelsize"] = 11


def run_eda(df: pd.DataFrame, meta: Dict[str, Any], out_dir: str = "outputs/figures") -> None:
    """
    Complete Exploratory Data Analysis for Instagram Insights.
    Generates high-quality charts + actionable insights for a social media agency.
    All files saved in 'outputs/figures/' (works perfectly with Streamlit).
    """

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    date_col = meta.get("date_col")
    reach_col = meta.get("reach_col")
    views_col = meta.get("views_col")
    er_col = meta.get("engagement_rate_col", "engagement_rate_pct")
    total_eng_col = meta.get("engagement_col", "total_engagements")
    denominator_col = reach_col or views_col or "impressions"

    print("\n" + "=" * 70)
    print("       INSTAGRAM PERFORMANCE - EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    # ===================================================================
    # 1. Dataset Overview
    # ===================================================================
    print(f"\n1. Dataset Overview")
    print(f"   • Total posts analyzed       : {len(df)}")
    if date_col:
        print(f"   • Date range                 : {df[date_col].min().date()} → {df[date_col].max().date()}")
    print(f"   • Engagement denominator used: {meta.get('engagement_denominator', 'unknown').upper()}")

    # ===================================================================
    # 2. Summary Statistics with Quartiles & IQR
    # ===================================================================
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) > 0:
        stats = df[numeric_cols].describe(percentiles=[.25, .5, .75]).T
        stats["IQR"] = stats["75%"] - stats["25%"]
        stats["lower_bound"] = stats["25%"] - 1.5 * stats["IQR"]
        stats["upper_bound"] = stats["75%"] + 1.5 * stats["IQR"]
        stats["outliers_count"] = (
            (df[numeric_cols] < stats.loc[numeric_cols, "lower_bound"]) |
            (df[numeric_cols] > stats.loc[numeric_cols, "upper_bound"])
        ).sum()
        stats = stats.round(2)

        print(f"\n2. Summary Statistics (Mean, Quartiles, IQR & Outliers)")
        print(stats[["mean", "25%", "50%", "75%", "IQR", "outliers_count"]])
        stats.to_csv(out_path / "01_summary_statistics_with_quartiles.csv", index=True)

    # ===================================================================
    # 3. Engagement Rate Analysis + Outliers (IQR Method)
    # ===================================================================
    if er_col in df.columns:
        Q1 = df[er_col].quantile(0.25)
        Q3 = df[er_col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[er_col] < lower) | (df[er_col] > upper)]

        print(f"\n3. Engagement Rate Outlier Analysis (IQR × 1.5)")
        print(f"   • Q1 (25%)           : {Q1:.2f}%")
        print(f"   • Median (50%)       : {df[er_col].median():.2f}%")
        print(f"   • Q3 (75%)           : {Q3:.2f}%")
        print(f"   • IQR                : {IQR:.2f}%")
        print(f"   • Outlier thresholds : < {lower:.2f}% or > {upper:.2f}%")
        print(f"   • Outliers detected  : {len(outliers)} posts ({len(outliers)/len(df)*100:.1f}% of total)")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        sns.boxplot(x=df[er_col], ax=ax1, color="#1DA1F2")
        ax1.set_title("Engagement Rate - Boxplot (IQR Method)")
        ax1.set_xlabel("Engagement Rate (%)")

        sns.histplot(df[er_col], kde=True, bins=30, color="#1877F2", ax=ax2)
        ax2.axvline(df[er_col].mean(), color="red", linestyle="--", label=f"Mean: {df[er_col].mean():.2f}%")
        ax2.axvline(df[er_col].median(), color="orange", linestyle="--", label=f"Median: {df[er_col].median():.2f}%")
        ax2.legend()
        ax2.set_title("Engagement Rate Distribution")
        ax2.set_xlabel("Engagement Rate (%)")

        plt.tight_layout()
        plt.savefig(out_path / "02_engagement_rate_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()

    # ===================================================================
    # 4. Best Posting Day & Hour
    # ===================================================================
    best_day = "N/A"
    top3_hours = ["N/A"]

    if date_col and pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df["weekday"] = df[date_col].dt.day_name()
        df["hour"] = df[date_col].dt.hour.astype(str) + ":00"

        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        plt.figure()
        sns.barplot(
            data=df, x="weekday", y=er_col, order=weekday_order,
            hue="weekday", palette="viridis", legend=False
        )
        plt.title("Average Engagement Rate by Day of the Week", pad=20)
        plt.ylabel("Engagement Rate (%)")
        plt.xlabel("")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(out_path / "03_best_day_to_post.png", dpi=300, bbox_inches="tight")
        plt.close()

        best_day = df.groupby("weekday")[er_col].mean().idxmax()
        print(f"   AGENCY RECOMMENDATION → BEST DAY TO POST: {best_day.upper()}")

        hourly = df.groupby("hour")[er_col].mean().sort_values(ascending=False)
        top3_hours = hourly.head(3).index.tolist()
        print(f"   Top 3 performing hours: {', '.join(top3_hours)}")

        plt.figure()
        hourly_df = df.groupby("hour")[er_col].mean().reset_index()
        sns.lineplot(data=hourly_df, x="hour", y=er_col, marker="o", color="#1DA1F2")
        plt.title("Engagement Rate by Hour of Day (24h)", pad=20)
        plt.xlabel("Hour")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(out_path / "04_best_hour_to_post.png", dpi=300, bbox_inches="tight")
        plt.close()

    # ===================================================================
    # 5. Reach/Views vs Engagement Rate
    # ===================================================================
    if denominator_col in df.columns and er_col in df.columns:
        plt.figure()
        sns.scatterplot(data=df, x=denominator_col, y=er_col, alpha=0.6, color="#1877F2")
        sns.regplot(data=df, x=denominator_col, y=er_col, scatter=False, color="red")
        plt.title(f"{denominator_col.replace('_', ' ').title()} vs Engagement Rate", pad=20)
        plt.xlabel(denominator_col.replace("_", " ").title())
        plt.ylabel("Engagement Rate (%)")
        plt.tight_layout()
        plt.savefig(out_path / "05_reach_vs_engagement.png", dpi=300, bbox_inches="tight")
        plt.close()

    # ===================================================================
    # 6. Top 10 Best Performing Posts
    # ===================================================================
    if er_col in df.columns:
        cols_to_save = [date_col, er_col, total_eng_col, denominator_col] if date_col else [er_col, total_eng_col, denominator_col]
        cols_to_save = [c for c in cols_to_save if c in df.columns]
        top10 = df.nlargest(10, er_col)[cols_to_save]
        top10.to_csv(out_path / "top10_best_posts.csv", index=False)
        print(f"\n   → Top 10 best-performing posts exported for content review!")

    # ===================================================================
    # 7. Final Actionable Insights
    # ===================================================================
    print("\n" + "=" * 70)
    print("ACTIONABLE RECOMMENDATIONS FOR THE AGENCY")
    print("=" * 70)
    print(f"• Current Average Engagement Rate : {df[er_col].mean():.2f}%")
    print(f"• Best day to post                : {best_day.upper()}")
    print(f"• Golden hours                    : {', '.join(top3_hours)}")
    print(f"• Viral posts (outliers)          : Study these {len(outliers) if 'outliers' in locals() else 0} posts to replicate success!")
    print(f"• All charts & CSVs saved in      : {out_path.resolve()}")
    print("=" * 70)
    print("EDA completed – Ready for Hypothesis Testing & Streamlit App!\n")


if __name__ == "__main__":
    from load_data import load_instagram_csv
    from preprocessing import preprocess_df

    df_raw = load_instagram_csv("InstagramData.csv")
    df_clean, meta = preprocess_df(df_raw)
    run_eda(df_clean, meta)