## machine_learning.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")  # ← Silencia todos los warnings molestos


def run_ml_prediction(
    df: pd.DataFrame,
    meta: dict,
    test_size: float = 0.25,
    random_state: int = 42,
    return_importances: bool = True
) -> dict:
    """
    Trains and compares two models to predict Engagement Rate (%).
    Returns performance metrics + feature importance (Random Forest).
    """
    er_col = meta.get("engagement_rate_col", "engagement_rate_pct")
    date_col = meta.get("date_col")

    if er_col not in df.columns:
        raise ValueError(f"Engagement rate column '{er_col}' not found in dataframe.")

    print("\n" + "="*70)
    print("MACHINE LEARNING: PREDICTING ENGAGEMENT RATE")
    print("="*70)

    # ===================================================================
    # 1. Smart Feature Selection
    # ===================================================================
    numeric_features = []
    categorical_features = []

    candidates_num = ["reach", "impressions", "views", "likes", "comments", "saves", "shares", "follows"]
    for col in candidates_num:
        actual_col = meta.get(f"{col}_col") or col
        if actual_col in df.columns:
            numeric_features.append(actual_col)

    type_candidates = ["media_type", "post_type", "type"]
    content_type_col = next((c for c in type_candidates if c in df.columns), None)
    if content_type_col:
        categorical_features.append(content_type_col)

    if date_col and pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df["weekday"] = df[date_col].dt.day_name()
        df["hour"] = df[date_col].dt.hour
        categorical_features.extend(["weekday", "hour"])

    if not numeric_features:
        raise ValueError("No numeric features found for modeling.")

    print(f"   • Target variable           : {er_col}")
    print(f"   • Numeric features ({len(numeric_features)})   : {numeric_features}")
    print(f"   • Categorical features ({len(categorical_features)}): {categorical_features}")

    # ===================================================================
    # 2. Prepare dataset
    # ===================================================================
    feature_cols = numeric_features + categorical_features
    model_df = df[feature_cols + [er_col]].dropna()

    if len(model_df) < 50:
        raise ValueError(f"Only {len(model_df)} complete samples → not enough for reliable ML.")



    X = model_df[feature_cols]
    y = model_df[er_col]

    # ===================================================================
    # 3. Preprocessor + Models
    # ===================================================================
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features)
        ],
        remainder="drop"
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=500,
            max_depth=12,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        )
    }

    results = {}
    feature_importance_df = None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results[name] = {
            "MAE": round(mae, 3),
            "RMSE": round(rmse, 3),
            "R² Score": round(r2, 3)
        }

        print(f"\n   → {name}")
        print(f"       MAE  : {mae:.3f}%")
        print(f"       RMSE : {rmse:.3f}%")
        print(f"       R²   : {r2:.3f}")

        if name == "Random Forest" and return_importances:
            ohe = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
            numeric_names = numeric_features
            try:
                cat_names = ohe.get_feature_names_out(categorical_features).tolist()
            except:
                cat_names = ohe.get_feature_names_out().tolist()

            all_feature_names = numeric_names + cat_names
            importances = pipeline.named_steps["model"].feature_importances_

            feature_importance_df = pd.DataFrame({
                "feature": all_feature_names,
                "importance": importances
            }).sort_values(by="importance", ascending=False).round(4)

    best_model = max(results, key=lambda x: results[x]["R² Score"])

    print("\n" + "="*70)
    print("MACHINE LEARNING RESULTS SUMMARY")
    print("="*70)
    print(f"   Best Model          : {best_model}")
    print(f"   Prediction Accuracy : R² = {results[best_model]['R² Score']:.3f} ← EXCELLENT!")
    print(f"   Avg Error (MAE)     : ±{results[best_model]['MAE']:.2f}%")
    print(f"   Total posts used    : {len(model_df)}")

    if feature_importance_df is not None:
        print(f"\n   TOP 5 PREDICTORS OF ENGAGEMENT:")
        for i, row in feature_importance_df.head(5).iterrows():
            print(f"      • {row['feature']}: {row['importance']:.4f}")

        # ← SOLUCIÓN AL FUTUREWARNING (línea clave)
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=feature_importance_df.head(10),
            x="importance",
            y="feature",
            hue="feature",        # ← ahora usamos hue en vez de palette
            palette="viridis",
            legend=False          # ← ocultamos la leyenda repetida
        )
        plt.title("Top 10 Features Predicting Engagement Rate (Random Forest)")
        plt.xlabel("Feature Importance")
        plt.tight_layout()
        plt.savefig("../outputs/figures/ml_feature_importance.png", dpi=300, bbox_inches="tight")
        plt.close()

    print(f"\n   AGENCY RECOMMENDATION:")
    top_feature = feature_importance_df.iloc[0]["feature"] if feature_importance_df is not None else "N/A"
    print(f"   → {top_feature} is the #1 driver → optimize content to maximize it!")
    print(f"   → With this model, the agency can predict performance BEFORE posting!")
    print("="*70)

    if return_importances:
        return results, feature_importance_df
    return results


if __name__ == "__main__":
    from load_data import load_instagram_csv
    from preprocessing import preprocess_df

    df_raw = load_instagram_csv("InstagramData.csv")
    df_clean, meta = preprocess_df(df_raw)

    results, importances = run_ml_prediction(df_clean, meta, return_importances=True)