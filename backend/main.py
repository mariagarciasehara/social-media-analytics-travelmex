# backend/main.py — Travel Mex Tours | FastAPI Backend v2
"""
REST API for Instagram analytics — updated with Performance Score as primary metric.

Endpoints:
    GET  /health      → service health check
    POST /upload      → upload CSV, get preview + metadata including performance scores
    POST /eda         → run EDA, returns Performance Score + ER insights
    POST /hypothesis  → run statistical tests on both Score and ER
    POST /ml          → train models predicting both Score and ER
    POST /predict     → predict Score + ER for a new post with interpretation
"""
import sys
import io
import warnings
import traceback
warnings.filterwarnings("ignore")
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from load_data import load_instagram_csv
from preprocessing import preprocess_df

# ── App setup ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Travel Mex — Instagram Analytics API v2",
    description=(
        "Data Science backend for Instagram performance analysis. "
        "Performance Score (0-100) is the primary metric — corrects for reach bias. "
        "Formula: 40% ER + 40% Total Engagements + 20% Reach."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helper: JSON-safe serialization ───────────────────────────────────────
def _safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_safe(i) for i in obj]
    if isinstance(obj, float):
        if obj != obj or obj == float("inf") or obj == float("-inf"):
            return None
        return round(obj, 4)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else round(float(obj), 4)
    if isinstance(obj, np.ndarray):
        return _safe(obj.tolist())
    if isinstance(obj, pd.Timestamp):
        return str(obj.date())
    if isinstance(obj, pd.Categorical):
        return str(obj)
    return obj


# ── Pydantic model for /predict ────────────────────────────────────────────
class PredictRequest(BaseModel):
    reach:     Optional[float] = None
    views:     Optional[float] = None
    likes:     Optional[float] = None
    comments:  Optional[float] = None
    saves:     Optional[float] = None
    shares:    Optional[float] = None
    follows:   Optional[float] = None
    post_type: Optional[str]   = None
    weekday:   Optional[str]   = None
    time_slot: Optional[str]   = None
    month:     Optional[str]   = None


# ══════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════

# ── GET /health ────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health():
    return {
        "status":  "ok",
        "service": "Travel Mex Analytics API",
        "version": "2.0.0",
        "primary_metric": "performance_score (0-100)",
        "formula": "40% ER + 40% Total Engagements + 20% Reach"
    }


# ── POST /upload ───────────────────────────────────────────────────────────
@app.post("/upload", tags=["Data"])
async def upload(file: UploadFile = File(...)):
    """
    Upload an Instagram CSV export.
    Returns shape, columns, full metadata including performance scores,
    tier distribution, and first 5 rows preview.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    try:
        raw      = await file.read()
        df_raw   = load_instagram_csv(io.BytesIO(raw))
        df, meta = preprocess_df(df_raw)

        preview = df.head(5).copy()
        for col in preview.select_dtypes(include=["datetime64[ns]"]).columns:
            preview[col] = preview[col].astype(str)
        for col in preview.select_dtypes(include=["category"]).columns:
            preview[col] = preview[col].astype(str)

        return JSONResponse(_safe({
            "status":   "success",
            "filename": file.filename,
            "shape":    {"rows": len(df), "cols": len(df.columns)},
            "columns":  df.columns.tolist(),
            "meta":     meta,
            "performance_summary": {
                "avg_score":         meta.get("avg_performance_score"),
                "high_performers":   meta.get("high_performers_count"),
                "medium_performers": meta.get("medium_performers_count"),
                "low_performers":    meta.get("low_performers_count"),
                "above_avg":         meta.get("above_avg_count"),
                "avg_relative":      meta.get("avg_count"),
                "below_avg":         meta.get("below_avg_count"),
                "interpretation": (
                    f"Posts use on average {meta.get('avg_performance_score',0):.1f}% "
                    "of the account's proven maximum potential."
                )
            },
            "preview": preview.to_dict(orient="records"),
        }))

    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


# ── POST /eda ──────────────────────────────────────────────────────────────
@app.post("/eda", tags=["Analysis"])
async def eda(file: UploadFile = File(...)):
    """
    Run full EDA. Returns Performance Score stats, ER stats, best timing,
    best content type, engagement breakdown, monthly trend, and top 10 posts.
    Primary metric: Performance Score. Secondary: Engagement Rate.
    """
    try:
        raw      = await file.read()
        df_raw   = load_instagram_csv(io.BytesIO(raw))
        df, meta = preprocess_df(df_raw)

        er_col        = meta["engagement_rate_col"]
        ps_col        = meta["performance_score_col"]
        tier_col      = meta["performance_tier_col"]
        tier_rel      = meta["performance_tier_relative_col"]
        type_col      = meta.get("type_col")
        time_slot_col = meta.get("time_slot_col")
        date_col      = meta.get("date_col")

        WEEKDAY_ORDER = ["Monday","Tuesday","Wednesday","Thursday",
                         "Friday","Saturday","Sunday"]

        er = df[er_col].dropna()
        ps = df[ps_col].dropna()

        # ── Performance Score stats ────────────────────────────────────────
        ps_stats = {
            "mean":   round(ps.mean(), 2),
            "median": round(ps.median(), 2),
            "std":    round(ps.std(), 2),
            "min":    round(ps.min(), 2),
            "max":    round(ps.max(), 2),
            "interpretation": (
                f"Posts use on average {ps.mean():.1f}/100 of the account's proven potential. "
                f"Best post reached {ps.max():.1f}/100."
            )
        }

        # ── ER stats ──────────────────────────────────────────────────────
        er_stats = {
            "mean":      round(er.mean(), 2),
            "median":    round(er.median(), 2),
            "std":       round(er.std(), 2),
            "min":       round(er.min(), 2),
            "max":       round(er.max(), 2),
            "benchmark": 5.0,
            "note":      "ER penalizes posts with higher reach — use Performance Score as primary metric."
        }

        # ── Best day ──────────────────────────────────────────────────────
        best_day_ps, best_day_er = None, None
        day_avgs_ps, day_avgs_er = {}, {}
        if "weekday" in df.columns:
            day_ps = df.groupby("weekday")[ps_col].mean()
            day_er = df.groupby("weekday")[er_col].mean()
            day_avgs_ps = {d: round(v, 2) for d, v in day_ps.items()}
            day_avgs_er = {d: round(v, 2) for d, v in day_er.items()}
            best_day_ps = day_ps.idxmax()
            best_day_er = day_er.idxmax()

        # ── Best time slot ────────────────────────────────────────────────
        best_slot_ps, slot_stats = None, {}
        if time_slot_col and time_slot_col in df.columns:
            slot_ps = df.groupby(time_slot_col, observed=True)[ps_col].mean()
            slot_er = df.groupby(time_slot_col, observed=True)[er_col].mean()
            slot_n  = df.groupby(time_slot_col, observed=True)[ps_col].count()
            best_slot_ps = slot_ps.idxmax()
            slot_stats = {
                str(s): {
                    "avg_score": round(float(slot_ps[s]), 2),
                    "avg_er":    round(float(slot_er[s]), 2),
                    "posts":     int(slot_n[s]),
                    "reliable":  bool(slot_n[s] >= 10)
                }
                for s in slot_ps.index
            }

        # ── Best content type ─────────────────────────────────────────────
        best_type_ps, best_type_er, type_stats = None, None, {}
        if type_col and type_col in df.columns:
            ts = df.groupby(type_col).agg(
                avg_score=(ps_col, "mean"),
                avg_er=(er_col, "mean"),
                posts=(er_col, "count")
            ).round(2)
            type_stats = {
                str(k): {
                    "avg_score": float(v["avg_score"]),
                    "avg_er":    float(v["avg_er"]),
                    "posts":     int(v["posts"])
                }
                for k, v in ts.iterrows()
            }
            best_type_ps = ts["avg_score"].idxmax()
            best_type_er = ts["avg_er"].idxmax()

        # ── Engagement breakdown ──────────────────────────────────────────
        breakdown = {}
        for label, col_key in [("likes","likes_col"),("comments","comments_col"),
                                ("saves","saves_col"),("shares","shares_col")]:
            col = meta.get(col_key)
            if col and col in df.columns:
                breakdown[label] = int(df[col].sum())

        # ── Monthly trend ─────────────────────────────────────────────────
        monthly_trend = []
        if date_col and pd.api.types.is_datetime64_any_dtype(df[date_col]):
            monthly_ps_s = df.set_index(date_col)[ps_col].resample("ME").mean().dropna()
            monthly_er_s = df.set_index(date_col)[er_col].resample("ME").mean().dropna()
            monthly_trend = [
                {
                    "month":     str(k.date()),
                    "avg_score": round(float(monthly_ps_s[k]), 2),
                    "avg_er":    round(float(monthly_er_s.get(k, np.nan)), 2)
                }
                for k in monthly_ps_s.index
            ]

        # ── Weekend check ─────────────────────────────────────────────────
        weekend_info = {}
        if "is_weekend" in df.columns:
            weekend_n = int((df["is_weekend"]==True).sum())
            weekend_info = {
                "weekend_posts":     weekend_n,
                "sufficient_data":   weekend_n >= 5,
                "recommendation":    (
                    "Weekend vs Weekday test available." if weekend_n >= 5
                    else f"Only {weekend_n} weekend post(s). Need ≥5 to activate this analysis."
                )
            }

        # ── Top 10 posts ──────────────────────────────────────────────────
        top10_cols = [c for c in [date_col, ps_col, er_col, tier_col, tier_rel,
                                   type_col, meta.get("likes_col"), meta.get("saves_col")]
                      if c and c in df.columns]
        top10 = df.nlargest(10, ps_col)[top10_cols].copy()
        if date_col in top10.columns:
            top10[date_col] = top10[date_col].astype(str)
        for col in top10.select_dtypes(include=["category"]).columns:
            top10[col] = top10[col].astype(str)

        return JSONResponse(_safe({
            "status":        "success",
            "total_posts":   len(df),
            "date_range":    {
                "start": meta.get("date_range_start"),
                "end":   meta.get("date_range_end")
            },
            "performance_score": ps_stats,
            "engagement_rate":   er_stats,
            "tier_distribution": {
                "vs_potential": {
                    "High":   meta.get("high_performers_count"),
                    "Medium": meta.get("medium_performers_count"),
                    "Low":    meta.get("low_performers_count"),
                },
                "vs_peers": {
                    "Above Average": meta.get("above_avg_count"),
                    "Average":       meta.get("avg_count"),
                    "Below Average": meta.get("below_avg_count"),
                }
            },
            "best_day_by_score":    best_day_ps,
            "best_day_by_er":       best_day_er,
            "day_averages_score":   day_avgs_ps,
            "day_averages_er":      day_avgs_er,
            "best_slot_by_score":   str(best_slot_ps) if best_slot_ps else None,
            "time_slot_stats":      slot_stats,
            "best_type_by_score":   str(best_type_ps) if best_type_ps else None,
            "best_type_by_er":      str(best_type_er) if best_type_er else None,
            "content_type_stats":   type_stats,
            "engagement_breakdown": breakdown,
            "monthly_trend":        monthly_trend,
            "weekend_info":         weekend_info,
            "top10_posts":          top10.to_dict(orient="records"),
            "benchmark_er":         5.0,
        }))

    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


# ── POST /hypothesis ───────────────────────────────────────────────────────
@app.post("/hypothesis", tags=["Analysis"])
async def hypothesis(file: UploadFile = File(...)):
    """
    Run hypothesis tests on both Performance Score and Engagement Rate.
    Tests: content type (ANOVA), time slot (ANOVA), weekend vs weekday (t-test).
    """
    from scipy import stats as scipy_stats

    try:
        raw      = await file.read()
        df_raw   = load_instagram_csv(io.BytesIO(raw))
        df, meta = preprocess_df(df_raw)

        er_col        = meta["engagement_rate_col"]
        ps_col        = meta["performance_score_col"]
        type_col      = meta.get("type_col")
        time_slot_col = meta.get("time_slot_col")
        ALPHA         = 0.05

        results = {}

        def eta2(groups):
            all_d = np.concatenate(groups)
            ss_b  = sum(len(g)*(g.mean()-all_d.mean())**2 for g in groups)
            ss_t  = sum((x-all_d.mean())**2 for g in groups for x in g)
            return round(ss_b/ss_t, 4) if ss_t > 0 else 0.0

        def effect_label(e):
            if e < 0.01: return "negligible"
            if e < 0.06: return "small"
            if e < 0.14: return "medium"
            return "large"

        def run_test(groups_dict, alpha=ALPHA):
            groups = list(groups_dict.values())
            names  = list(groups_dict.keys())
            if len(groups) == 2:
                stat, p   = scipy_stats.ttest_ind(groups[0], groups[1], equal_var=False)
                test_name = "Welch t-test"
            else:
                stat, p   = scipy_stats.f_oneway(*groups)
                test_name = "One-Way ANOVA"
            e2     = eta2(groups)
            reject = p < alpha
            best   = str(names[int(np.argmax([g.mean() for g in groups]))])
            return {
                "test":         test_name,
                "statistic":    round(float(stat), 4),
                "p_value":      round(float(p), 4),
                "reject_H0":    reject,
                "effect_size":  e2,
                "effect_label": effect_label(e2),
                "best_group":   best,
                "group_means":  {str(n): round(float(g.mean()), 4)
                                  for n, g in zip(names, groups)},
                "significant":  reject,
            }

        # Test 1: Content Type
        if type_col and type_col in df.columns:
            groups_ps = {name: grp[ps_col].dropna().values
                         for name, grp in df.groupby(type_col)
                         if len(grp[ps_col].dropna()) >= 5}
            groups_er = {name: grp[er_col].dropna().values
                         for name, grp in df.groupby(type_col)
                         if len(grp[er_col].dropna()) >= 5}

            if len(groups_ps) >= 2:
                r_ps = run_test(groups_ps)
                r_er = run_test(groups_er)
                results["content_type"] = {
                    "question":          "Does content type significantly affect performance?",
                    "by_performance_score": r_ps,
                    "by_engagement_rate":   r_er,
                    "metrics_agree":     r_ps["best_group"] == r_er["best_group"],
                    "recommendation": (
                        f"Both metrics agree: prioritize {r_ps['best_group']}."
                        if r_ps["best_group"] == r_er["best_group"]
                        else f"Metrics differ. Score → {r_ps['best_group']} | ER → {r_er['best_group']}. "
                             f"Use Performance Score as primary: prioritize {r_ps['best_group']}."
                    )
                }

        # Test 2: Time Slot
        if time_slot_col and time_slot_col in df.columns:
            slot_groups_ps = {str(name): grp[ps_col].dropna().values
                              for name, grp in df.groupby(time_slot_col, observed=True)
                              if len(grp[ps_col].dropna()) >= 5}
            if len(slot_groups_ps) >= 2:
                r_slot = run_test(slot_groups_ps)
                results["time_slot"] = {
                    "question":             "Does time slot significantly affect Performance Score?",
                    "by_performance_score": r_slot,
                    "statistical_note":     "Early Morning has ~89 posts vs ~16 in other slots. Results are directional.",
                    "recommendation": (
                        f"Best slot: {r_slot['best_group']}. "
                        "Run experiment posting 10x in Late Morning and Afternoon to confirm."
                    )
                }

        # Test 3: Weekend vs Weekday
        if "is_weekend" in df.columns:
            weekend_ps = df[df["is_weekend"]==True][ps_col].dropna().values
            weekday_ps = df[df["is_weekend"]==False][ps_col].dropna().values
            weekend_er = df[df["is_weekend"]==True][er_col].dropna().values
            weekday_er = df[df["is_weekend"]==False][er_col].dropna().values

            if len(weekend_ps) < 5:
                results["weekend"] = {
                    "status":          "insufficient_data",
                    "weekend_posts":   int(len(weekend_ps)),
                    "minimum_needed":  5,
                    "recommendation":  (
                        f"Only {len(weekend_ps)} weekend post(s). "
                        "Post on weekends for 2-3 months to activate this test."
                    )
                }
            else:
                r_ps = run_test({"Weekday": weekday_ps, "Weekend": weekend_ps})
                r_er = run_test({"Weekday": weekday_er, "Weekend": weekend_er})
                results["weekend"] = {
                    "question":             "Do weekend posts outperform weekday posts?",
                    "by_performance_score": r_ps,
                    "by_engagement_rate":   r_er,
                    "weekday_avg_score":    round(float(weekday_ps.mean()), 2),
                    "weekend_avg_score":    round(float(weekend_ps.mean()), 2),
                }

        return JSONResponse(_safe({
            "status":  "success",
            "alpha":   ALPHA,
            "results": results
        }))

    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


# ── POST /ml ───────────────────────────────────────────────────────────────
@app.post("/ml", tags=["Machine Learning"])
async def ml(file: UploadFile = File(...)):
    """
    Train ML models predicting both Performance Score (primary) and ER (secondary).
    Returns model metrics, feature importance for both targets, and summary.
    """
    from sklearn.model_selection import train_test_split, KFold, cross_val_score
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    try:
        raw      = await file.read()
        df_raw   = load_instagram_csv(io.BytesIO(raw))
        df, meta = preprocess_df(df_raw)

        er_col        = meta["engagement_rate_col"]
        ps_col        = meta["performance_score_col"]
        type_col      = meta.get("type_col")
        time_slot_col = meta.get("time_slot_col")

        num_keys  = ["reach_col","views_col","likes_col","comments_col",
                     "saves_col","shares_col","follows_col"]
        num_feats = [meta[k] for k in num_keys if meta.get(k) and meta[k] in df.columns]
        cat_feats = []
        if type_col and type_col in df.columns:            cat_feats.append(type_col)
        if "weekday" in df.columns:                        cat_feats.append("weekday")
        if time_slot_col and time_slot_col in df.columns:  cat_feats.append(time_slot_col)

        all_feats = num_feats + cat_feats
        mdf       = df[all_feats + [ps_col, er_col]].dropna()

        if len(mdf) < 30:
            raise HTTPException(status_code=422,
                detail=f"Only {len(mdf)} complete rows — need ≥30.")

        X    = mdf[all_feats]
        y_ps = mdf[ps_col]
        y_er = mdf[er_col]

        X_train, X_test, y_train_ps, y_test_ps = train_test_split(
            X, y_ps, test_size=0.2, random_state=42)
        _, _, y_train_er, y_test_er = train_test_split(
            X, y_er, test_size=0.2, random_state=42)

        transformers = [("num", StandardScaler(), num_feats)]
        if cat_feats:
            transformers.append(
                ("cat", OneHotEncoder(drop="first", handle_unknown="ignore",
                                      sparse_output=False), cat_feats))
        pre = ColumnTransformer(transformers, remainder="drop")
        cv  = KFold(n_splits=5, shuffle=True, random_state=42)

        model_zoo = {
            "Linear Regression":  LinearRegression(),
            "Ridge Regression":   Ridge(alpha=1.0),
            "Random Forest":      RandomForestRegressor(
                n_estimators=400, max_depth=10, min_samples_leaf=3,
                random_state=42, n_jobs=-1),
            "Gradient Boosting":  GradientBoostingRegressor(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, random_state=42),
        }

        def train_all(X, X_train, X_test, y_train, y_test, y_full):
            results, pipes = {}, {}
            for name, model in model_zoo.items():
                pipe = Pipeline([("pre", pre), ("model", model)])
                pipe.fit(X_train, y_train)
                y_pred    = pipe.predict(X_test)
                cv_scores = cross_val_score(pipe, X, y_full, cv=cv,
                                            scoring="r2", n_jobs=-1)
                results[name] = {
                    "R2":        round(float(r2_score(y_test, y_pred)), 4),
                    "MAE":       round(float(mean_absolute_error(y_test, y_pred)), 4),
                    "RMSE":      round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
                    "CV_R2":     round(float(cv_scores.mean()), 4),
                    "CV_R2_std": round(float(cv_scores.std()), 4),
                }
                pipes[name] = pipe
            best = max(results, key=lambda x: results[x]["R2"])
            return results, pipes, best

        results_ps, pipes_ps, best_ps = train_all(
            X, X_train, X_test, y_train_ps, y_test_ps, y_ps)
        results_er, pipes_er, best_er = train_all(
            X, X_train, X_test, y_train_er, y_test_er, y_er)

        def get_fi(pipes, best):
            for name in [best, "Gradient Boosting", "Random Forest"]:
                if name not in pipes: continue
                pipe   = pipes[name]
                fitted = pipe.named_steps["pre"]
                try:
                    ohe       = fitted.named_transformers_["cat"]
                    cat_names = ohe.get_feature_names_out(cat_feats).tolist()
                except Exception:
                    cat_names = []
                all_names   = num_feats + cat_names
                importances = pipe.named_steps["model"].feature_importances_
                if len(importances) == len(all_names):
                    fi = sorted(
                        [{"feature": n, "importance": round(float(i), 4)}
                         for n, i in zip(all_names, importances)],
                        key=lambda x: -x["importance"]
                    )
                    return fi[:10]
            return []

        fi_ps = get_fi(pipes_ps, best_ps)
        fi_er = get_fi(pipes_er, best_er)

        bm_ps    = results_ps[best_ps]
        bm_er    = results_er[best_er]
        top_ps   = fi_ps[0]["feature"] if fi_ps else "N/A"
        top_er   = fi_er[0]["feature"] if fi_er else "N/A"

        return JSONResponse(_safe({
            "status":            "success",
            "total_posts_used":  len(mdf),
            "primary_target":    ps_col,
            "secondary_target":  er_col,
            "performance_score_models": {
                "results":            results_ps,
                "best_model":         best_ps,
                "feature_importance": fi_ps,
                "summary": (
                    f"Best: {best_ps} (R²={bm_ps['R2']}, "
                    f"MAE=±{bm_ps['MAE']} pts, CV R²={bm_ps['CV_R2']}). "
                    f"Top driver: {top_ps}."
                )
            },
            "engagement_rate_models": {
                "results":            results_er,
                "best_model":         best_er,
                "feature_importance": fi_er,
                "summary": (
                    f"Best: {best_er} (R²={bm_er['R2']}, "
                    f"MAE=±{bm_er['MAE']}%, CV R²={bm_er['CV_R2']}). "
                    f"Top driver: {top_er}."
                )
            },
            "drivers_agree":  top_ps == top_er,
            "recommendation": (
                f"Both targets share top driver: {top_ps}. Optimize this metric."
                if top_ps == top_er
                else f"Score driver: {top_ps} | ER driver: {top_er}. "
                     f"Optimizing for ER alone may not maximize overall impact."
            )
        }))

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


# ── POST /predict ──────────────────────────────────────────────────────────
@app.post("/predict", tags=["Machine Learning"])
async def predict(request: PredictRequest, file: UploadFile = File(...)):
    """
    Train models on uploaded data, then predict both Performance Score and ER
    for a new post. Returns scores, interpretation, and actionable recommendation.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import GradientBoostingRegressor

    try:
        raw      = await file.read()
        df_raw   = load_instagram_csv(io.BytesIO(raw))
        df, meta = preprocess_df(df_raw)

        er_col        = meta["engagement_rate_col"]
        ps_col        = meta["performance_score_col"]
        type_col      = meta.get("type_col")
        time_slot_col = meta.get("time_slot_col")

        num_keys  = ["reach_col","views_col","likes_col","comments_col",
                     "saves_col","shares_col","follows_col"]
        num_feats = [meta[k] for k in num_keys if meta.get(k) and meta[k] in df.columns]
        cat_feats = []
        if type_col and type_col in df.columns:            cat_feats.append(type_col)
        if "weekday" in df.columns:                        cat_feats.append("weekday")
        if time_slot_col and time_slot_col in df.columns:  cat_feats.append(time_slot_col)

        all_feats = num_feats + cat_feats
        mdf       = df[all_feats + [ps_col, er_col]].dropna()

        if len(mdf) < 20:
            raise HTTPException(status_code=422, detail="Not enough data to train model.")

        transformers = [("num", StandardScaler(), num_feats)]
        if cat_feats:
            transformers.append(
                ("cat", OneHotEncoder(drop="first", handle_unknown="ignore",
                                      sparse_output=False), cat_feats))
        pre    = ColumnTransformer(transformers, remainder="drop")
        params = dict(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42)

        pipe_ps = Pipeline([("pre", pre), ("model", GradientBoostingRegressor(**params))])
        pipe_ps.fit(mdf[all_feats], mdf[ps_col])

        pipe_er = Pipeline([("pre", pre), ("model", GradientBoostingRegressor(**params))])
        pipe_er.fit(mdf[all_feats], mdf[er_col])

        # Build input
        col_map = {
            "reach":    meta.get("reach_col"),
            "views":    meta.get("views_col"),
            "likes":    meta.get("likes_col"),
            "comments": meta.get("comments_col"),
            "saves":    meta.get("saves_col"),
            "shares":   meta.get("shares_col"),
            "follows":  meta.get("follows_col"),
        }
        new_post = {}
        for req_key, col_name in col_map.items():
            if col_name and col_name in all_feats:
                val = getattr(request, req_key, None)
                new_post[col_name] = val if val is not None else float(df[col_name].median())

        if type_col and type_col in cat_feats:
            new_post[type_col] = request.post_type or str(df[type_col].mode()[0])
        if "weekday" in cat_feats:
            new_post["weekday"] = request.weekday or "Tuesday"
        if time_slot_col and time_slot_col in cat_feats:
            new_post[time_slot_col] = (
                request.time_slot or
                str(df.groupby(time_slot_col, observed=True)[ps_col].mean().idxmax())
            )

        pred_ps  = float(max(0, pipe_ps.predict(pd.DataFrame([new_post]))[0]))
        pred_er  = float(max(0, pipe_er.predict(pd.DataFrame([new_post]))[0]))
        avg_ps   = float(df[ps_col].mean())
        avg_er   = float(df[er_col].mean())

        # Interpretation
        high_ps = pred_ps >= avg_ps
        high_er = pred_er >= avg_er
        if high_ps and high_er:
            scenario = "strong_post"
            recommendation = "Publish and consider boosting to maximize reach!"
        elif high_ps and not high_er:
            scenario = "good_reach"
            recommendation = "Good absolute impact — add a strong CTA to drive more interactions."
        elif not high_ps and high_er:
            scenario = "good_engagement"
            recommendation = "Good engagement quality but limited reach — consider boosting."
        else:
            scenario = "below_average"
            recommendation = "Review content type, timing, or caption before publishing."

        return JSONResponse(_safe({
            "status":            "success",
            "predicted_score":   round(pred_ps, 2),
            "predicted_er":      round(pred_er, 2),
            "your_avg_score":    round(avg_ps, 2),
            "your_avg_er":       round(avg_er, 2),
            "benchmark_er":      5.0,
            "above_avg_score":   high_ps,
            "above_avg_er":      high_er,
            "above_benchmark_er": pred_er >= 5.0,
            "scenario":          scenario,
            "recommendation":    recommendation,
        }))

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


# ── Run ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
