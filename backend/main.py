# backend/main.py — Travel Mex Tours | FastAPI Backend
"""
REST API that exposes Instagram analytics as endpoints.
Consumed by Streamlit frontend or any other client.

Endpoints:
    GET  /health      → service health check
    POST /upload      → upload CSV, get preview + metadata
    POST /eda         → run EDA, get insights + stats
    POST /hypothesis  → run statistical tests
    POST /ml          → train models, get metrics + feature importance
    POST /predict     → predict ER for a new post
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
    title="Travel Mex — Instagram Analytics API",
    description="Data Science backend for Instagram performance analysis.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helper: make any value JSON-safe ──────────────────────────────────────
def _safe(obj: Any) -> Any:
    """Recursively replace NaN/Inf/numpy types for JSON serialization."""
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
    return obj


# ── Pydantic model for /predict ────────────────────────────────────────────
class PredictRequest(BaseModel):
    reach:    Optional[float] = None
    views:    Optional[float] = None
    likes:    Optional[float] = None
    comments: Optional[float] = None
    saves:    Optional[float] = None
    shares:   Optional[float] = None
    follows:  Optional[float] = None
    post_type: Optional[str]  = None
    weekday:   Optional[str]  = None
    hour:      Optional[int]  = None


# ══════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════

# ── GET /health ────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health():
    """Quick health check — returns 200 if the API is running."""
    return {"status": "ok", "service": "Travel Mex Analytics API", "version": "1.0.0"}


# ── POST /upload ───────────────────────────────────────────────────────────
@app.post("/upload", tags=["Data"])
async def upload(file: UploadFile = File(...)):
    """
    Upload an Instagram CSV export.
    Returns: shape, column names, metadata, and first 5 rows preview.
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

        return JSONResponse(_safe({
            "status":   "success",
            "filename": file.filename,
            "shape":    {"rows": len(df), "cols": len(df.columns)},
            "columns":  df.columns.tolist(),
            "meta":     meta,
            "preview":  preview.to_dict(orient="records"),
        }))

    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


# ── POST /eda ──────────────────────────────────────────────────────────────
@app.post("/eda", tags=["Analysis"])
async def eda(file: UploadFile = File(...)):
    """
    Run full Exploratory Data Analysis on the uploaded CSV.
    Returns: summary stats, insights (best day, best hour, top content type),
             engagement rate stats, and top 10 posts.
    """
    try:
        raw      = await file.read()
        df_raw   = load_instagram_csv(io.BytesIO(raw))
        df, meta = preprocess_df(df_raw)

        er_col   = meta["engagement_rate_col"]
        type_col = meta.get("type_col")
        er       = df[er_col].dropna()

        WEEKDAY_ORDER = ["Monday","Tuesday","Wednesday","Thursday","Friday",
                         "Saturday","Sunday"]

        # ── Engagement rate stats ──────────────────────────────────────────
        er_stats = {
            "mean":   round(er.mean(), 2),
            "median": round(er.median(), 2),
            "std":    round(er.std(), 2),
            "min":    round(er.min(), 2),
            "max":    round(er.max(), 2),
            "q25":    round(er.quantile(0.25), 2),
            "q75":    round(er.quantile(0.75), 2),
        }

        # ── Best day ──────────────────────────────────────────────────────
        best_day = None
        day_avg  = {}
        if "weekday" in df.columns:
            day_series = df.groupby("weekday")[er_col].mean()
            day_avg    = {d: round(v, 2) for d, v in day_series.items()}
            best_day   = day_series.idxmax()

        # ── Best hour ─────────────────────────────────────────────────────
        best_hour  = None
        top3_hours = []
        if "hour" in df.columns:
            hour_series = df.groupby("hour")[er_col].mean()
            best_hour   = int(hour_series.idxmax())
            top3_hours  = [int(h) for h in hour_series.nlargest(3).index.tolist()]

        # ── Best content type ─────────────────────────────────────────────
        best_type   = None
        type_stats  = {}
        if type_col and type_col in df.columns:
            ts = df.groupby(type_col)[er_col].agg(["mean","median","count"])
            type_stats = {
                str(k): {"avg_er": round(v["mean"], 2),
                         "median_er": round(v["median"], 2),
                         "posts": int(v["count"])}
                for k, v in ts.iterrows()
            }
            best_type = ts["mean"].idxmax()

        # ── Engagement breakdown ──────────────────────────────────────────
        breakdown = {}
        for label, col_key in [("likes","likes_col"),("comments","comments_col"),
                                ("saves","saves_col"),("shares","shares_col")]:
            col = meta.get(col_key)
            if col and col in df.columns:
                breakdown[label] = int(df[col].sum())

        # ── Monthly trend ─────────────────────────────────────────────────
        monthly_trend = []
        date_col = meta.get("date_col")
        if date_col and pd.api.types.is_datetime64_any_dtype(df[date_col]):
            monthly = df.set_index(date_col)[er_col].resample("ME").mean().dropna()
            monthly_trend = [
                {"month": str(k.date()), "avg_er": round(v, 2)}
                for k, v in monthly.items()
            ]

        # ── Top 10 posts ──────────────────────────────────────────────────
        top10_cols = [c for c in [date_col, er_col, type_col,
                                   meta.get("likes_col"), meta.get("saves_col")]
                      if c and c in df.columns]
        top10 = df.nlargest(10, er_col)[top10_cols].copy()
        if date_col in top10.columns:
            top10[date_col] = top10[date_col].astype(str)

        return JSONResponse(_safe({
            "status":          "success",
            "total_posts":     len(df),
            "date_range":      {"start": meta.get("date_range_start"),
                                "end":   meta.get("date_range_end")},
            "engagement_rate": er_stats,
            "best_day":        best_day,
            "day_averages":    day_avg,
            "best_hour":       best_hour,
            "top3_hours":      top3_hours,
            "best_content_type": str(best_type) if best_type else None,
            "content_type_stats": type_stats,
            "engagement_breakdown": breakdown,
            "monthly_trend":   monthly_trend,
            "top10_posts":     top10.to_dict(orient="records"),
            "benchmark":       5.0,
            "beats_benchmark": bool(er.mean() > 5.0),
        }))

    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


# ── POST /hypothesis ───────────────────────────────────────────────────────
@app.post("/hypothesis", tags=["Analysis"])
async def hypothesis(file: UploadFile = File(...)):
    """
    Run hypothesis tests on the uploaded CSV.
    Tests: content type vs ER (ANOVA), weekend vs weekday (t-test).
    """
    from scipy import stats as scipy_stats

    try:
        raw      = await file.read()
        df_raw   = load_instagram_csv(io.BytesIO(raw))
        df, meta = preprocess_df(df_raw)

        er_col   = meta["engagement_rate_col"]
        type_col = meta.get("type_col")
        ALPHA    = 0.05
        results  = {}

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

        # Test 1: Content type vs ER
        if type_col and type_col in df.columns:
            groups = {
                name: grp[er_col].dropna().values
                for name, grp in df.groupby(type_col)
                if len(grp[er_col].dropna()) >= 5
            }
            if len(groups) >= 2:
                f, p   = scipy_stats.f_oneway(*groups.values())
                e2     = eta2(list(groups.values()))
                reject = bool(p < ALPHA)
                best   = max(groups, key=lambda x: groups[x].mean())
                results["test_content_type"] = {
                    "question":    "Does content type significantly affect Engagement Rate?",
                    "test":        "One-Way ANOVA",
                    "statistic":   round(float(f), 4),
                    "p_value":     round(float(p), 4),
                    "reject_H0":   reject,
                    "effect_size": e2,
                    "effect_label": effect_label(e2),
                    "best_type":   str(best),
                    "group_means": {str(k): round(float(v.mean()), 2)
                                    for k, v in groups.items()},
                    "interpretation": (
                        f"Content type {'DOES' if reject else 'does NOT'} "
                        f"significantly affect ER (p={round(float(p),4)}, "
                        f"effect={effect_label(e2)}). Best: {best}."
                    )
                }

        # Test 2: Weekend vs Weekday
        if "is_weekend" in df.columns:
            weekend = df[df["is_weekend"]==True][er_col].dropna().values
            weekday = df[df["is_weekend"]==False][er_col].dropna().values
            if len(weekend) >= 5 and len(weekday) >= 5:
                t, p   = scipy_stats.ttest_ind(weekend, weekday, equal_var=False)
                e2     = eta2([weekend, weekday])
                reject = bool(p < ALPHA)
                better = "Weekend" if weekend.mean() > weekday.mean() else "Weekday"
                results["test_weekend"] = {
                    "question":    "Do weekend posts outperform weekday posts?",
                    "test":        "Welch t-test",
                    "statistic":   round(float(t), 4),
                    "p_value":     round(float(p), 4),
                    "reject_H0":   reject,
                    "effect_size": e2,
                    "effect_label": effect_label(e2),
                    "better_period": better,
                    "weekday_avg": round(float(weekday.mean()), 2),
                    "weekend_avg": round(float(weekend.mean()), 2),
                    "interpretation": (
                        f"{better} posts perform {'significantly' if reject else 'slightly'} "
                        f"better (p={round(float(p),4)}, effect={effect_label(e2)}). "
                        f"Weekday avg: {round(float(weekday.mean()),2)}% | "
                        f"Weekend avg: {round(float(weekend.mean()),2)}%."
                    )
                }

        return JSONResponse(_safe({"status": "success", "alpha": ALPHA, "results": results}))

    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


# ── POST /ml ───────────────────────────────────────────────────────────────
@app.post("/ml", tags=["Machine Learning"])
async def ml(file: UploadFile = File(...)):
    """
    Train and compare ML models to predict Engagement Rate.
    Returns: model metrics, best model, feature importance, summary.
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

        er_col   = meta["engagement_rate_col"]
        type_col = meta.get("type_col")

        # Feature selection
        num_keys  = ["reach_col","views_col","likes_col","comments_col",
                     "saves_col","shares_col","follows_col"]
        num_feats = [meta[k] for k in num_keys if meta.get(k) and meta[k] in df.columns]
        cat_feats = []
        if type_col and type_col in df.columns: cat_feats.append(type_col)
        if "weekday" in df.columns: cat_feats.append("weekday")
        if "hour"    in df.columns: cat_feats.append("hour")

        all_feats = num_feats + cat_feats
        mdf       = df[all_feats + [er_col]].dropna()

        if len(mdf) < 30:
            raise HTTPException(status_code=422,
                detail=f"Only {len(mdf)} complete rows — need ≥30.")

        X = mdf[all_feats]
        y = mdf[er_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

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

        results   = {}
        pipelines = {}

        for name, model in model_zoo.items():
            pipe = Pipeline([("pre", pre), ("model", model)])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring="r2", n_jobs=-1)
            results[name] = {
                "R2":          round(float(r2_score(y_test, y_pred)), 4),
                "MAE":         round(float(mean_absolute_error(y_test, y_pred)), 4),
                "RMSE":        round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
                "CV_R2_mean":  round(float(cv_scores.mean()), 4),
                "CV_R2_std":   round(float(cv_scores.std()), 4),
            }
            pipelines[name] = pipe

        best_model = max(results, key=lambda x: results[x]["R2"])

        # Feature importance
        feature_importance = []
        for fname in ["Gradient Boosting", "Random Forest"]:
            if fname in pipelines:
                pipe     = pipelines[fname]
                fitted   = pipe.named_steps["pre"]
                num_names = num_feats.copy()
                try:
                    ohe       = fitted.named_transformers_["cat"]
                    cat_names = ohe.get_feature_names_out(cat_feats).tolist()
                except Exception:
                    cat_names = []
                all_names   = num_names + cat_names
                importances = pipe.named_steps["model"].feature_importances_
                if len(importances) == len(all_names):
                    feature_importance = sorted(
                        [{"feature": n, "importance": round(float(i), 4)}
                         for n, i in zip(all_names, importances)],
                        key=lambda x: -x["importance"]
                    )
                break

        bm        = results[best_model]
        top_feat  = feature_importance[0]["feature"] if feature_importance else "N/A"
        summary   = (
            f"Best model: {best_model} "
            f"(R²={bm['R2']}, MAE=±{bm['MAE']}%, CV R²={bm['CV_R2_mean']}±{bm['CV_R2_std']}). "
            f"Top engagement driver: {top_feat}."
        )

        return JSONResponse(_safe({
            "status":             "success",
            "total_posts_used":   len(mdf),
            "model_results":      results,
            "best_model":         best_model,
            "feature_importance": feature_importance[:10],
            "numeric_features":   num_feats,
            "categorical_features": cat_feats,
            "summary":            summary,
        }))

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


# ── POST /predict ──────────────────────────────────────────────────────────
@app.post("/predict", tags=["Machine Learning"])
async def predict(request: PredictRequest, file: UploadFile = File(...)):
    """
    Train model on uploaded data, then predict ER for a new post.
    Send the CSV as a file + post details as JSON body.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import GradientBoostingRegressor

    try:
        raw      = await file.read()
        df_raw   = load_instagram_csv(io.BytesIO(raw))
        df, meta = preprocess_df(df_raw)

        er_col   = meta["engagement_rate_col"]
        type_col = meta.get("type_col")

        num_keys  = ["reach_col","views_col","likes_col","comments_col",
                     "saves_col","shares_col","follows_col"]
        num_feats = [meta[k] for k in num_keys if meta.get(k) and meta[k] in df.columns]
        cat_feats = []
        if type_col and type_col in df.columns: cat_feats.append(type_col)
        if "weekday" in df.columns: cat_feats.append("weekday")
        if "hour"    in df.columns: cat_feats.append("hour")

        all_feats = num_feats + cat_feats
        mdf       = df[all_feats + [er_col]].dropna()

        if len(mdf) < 20:
            raise HTTPException(status_code=422, detail="Not enough data to train model.")

        transformers = [("num", StandardScaler(), num_feats)]
        if cat_feats:
            transformers.append(
                ("cat", OneHotEncoder(drop="first", handle_unknown="ignore",
                                      sparse_output=False), cat_feats))

        pipe = Pipeline([
            ("pre", ColumnTransformer(transformers, remainder="drop")),
            ("model", GradientBoostingRegressor(
                n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42))
        ])
        pipe.fit(mdf[all_feats], mdf[er_col])

        # Build input from request
        col_map = {
            "reach": meta.get("reach_col"), "views": meta.get("views_col"),
            "likes": meta.get("likes_col"), "comments": meta.get("comments_col"),
            "saves": meta.get("saves_col"), "shares": meta.get("shares_col"),
            "follows": meta.get("follows_col"),
        }
        new_post = {}
        for req_key, col_name in col_map.items():
            if col_name and col_name in all_feats:
                val = getattr(request, req_key, None)
                new_post[col_name] = val if val is not None else df[col_name].median()

        if type_col and type_col in cat_feats:
            new_post[type_col] = request.post_type or df[type_col].mode()[0]
        if "weekday" in cat_feats:
            new_post["weekday"] = request.weekday or "Tuesday"
        if "hour" in cat_feats:
            new_post["hour"] = request.hour if request.hour is not None else 9

        prediction = float(max(0, pipe.predict(pd.DataFrame([new_post]))[0]))
        avg_er     = float(df[er_col].mean())

        return JSONResponse(_safe({
            "status":          "success",
            "predicted_er":    round(prediction, 2),
            "your_avg_er":     round(avg_er, 2),
            "benchmark":       5.0,
            "above_avg":       prediction >= avg_er,
            "above_benchmark": prediction >= 5.0,
            "recommendation":  (
                "Strong post — consider boosting to maximize reach!"
                if prediction >= avg_er
                else "Consider adjusting content type or posting time."
            )
        }))

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


# ── Run ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
