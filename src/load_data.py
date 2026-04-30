import pandas as pd
from pathlib import Path
from typing import Literal, Optional
import warnings

ContentType = Literal["posts", "reels", "stories", "videos", "unknown"]


def detect_content_type(df: pd.DataFrame) -> ContentType:
    """
    Automatically detects the content type based on characteristic columns.
    """
    cols = set(df.columns.str.lower())

    if "reach" in cols and "impressions" in cols:
        if "plays" in cols:
            return "reels"
        if "video_views" in cols:
            return "videos"
        if "replies" in cols or "exits" in cols:
            return "stories"
        return "posts"

    return "unknown"


def load_instagram_csv(
        path: str | Path,
        min_numeric_ratio: float = 0.6,
) -> pd.DataFrame:
    """
    Loads and cleans Instagram Insights CSV exports in a robust way.

    Handles:
    - Headers located in different rows
    - UTF-8 with/without BOM and other encodings
    - Bad lines and duplicate columns
    - Smart numeric and date conversion
    - Automatic content type detection (Posts, Reels, Stories, etc.)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Try common encodings (Instagram sometimes adds BOM)
    for encoding in ["utf-8-sig", "utf-8", "latin1"]:
        try:
            sample = pd.read_csv(path, nrows=10, header=None, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("Could not read the file with common encodings")

    # Detect the real header row (the one with most keyword matches)
    header_row_idx = 0
    max_score = 0
    keywords = ["id", "reach", "impressions", "likes", "comments", "date", "type", "posted", "published"]

    for i in range(min(5, len(sample))):
        row = sample.iloc[i].astype(str)
        score = sum(word in " ".join(row.str.lower().values) for word in keywords)
        if score > max_score:
            max_score = score
            header_row_idx = i

    # Load the full file skipping rows until the real header
    df_raw = pd.read_csv(
        path,
        skiprows=header_row_idx,
        encoding=encoding,
        dtype=str,
        low_memory=False,
        on_bad_lines="skip",
    )

    # Clean column names
    df_raw.columns = (
        df_raw.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^\w]+", "_", regex=True)
        .str.replace(r"^_+|_+$", "", regex=True)
        .str.replace(r"_+", "_", regex=True)
    )

    # Rename duplicate columns
    cols = pd.Series(df_raw.columns)
    for dup in df_raw.columns[df_raw.columns.duplicated(keep=False)]:
        count = sum(cols == dup)
        if count > 1:
            indices = cols[cols == dup].index
            for idx, seq in zip(indices[1:], range(1, count)):
                cols.iloc[idx] = f"{dup}_{seq}"
    df_raw.columns = cols

    df = df_raw.copy()

    # Keep ID columns as strings
    id_cols = [c for c in df.columns if "id" in c.lower()]
    for col in id_cols:
        df[col] = df[col].astype(str).str.replace(r"\.0$", "", regex=True)

    # Safe numeric conversion
    for col in df.columns:
        if col in id_cols:
            continue

        serie = df[col].astype(str).str.strip()

        # Clean common metric symbols (%, $, k, m)
        if serie.str.contains(r"[%,$kKmM]", regex=True).any():
            serie = serie.str.replace(r"[,%$]", "", regex=True)
            serie = serie.str.replace(r"^(\d+\.?\d*)k$", lambda m: str(float(m.group(1)) * 1000), regex=True)
            serie = serie.str.replace(r"^(\d+\.?\d*)m$", lambda m: str(float(m.group(1)) * 1000000), regex=True)

        numeric = pd.to_numeric(serie, errors="coerce")
        if numeric.notna().mean() >= min_numeric_ratio:
            df[col] = numeric

    # Final date conversion (silent, no warnings)
    date_cols = [c for c in df.columns if any(k in c.lower() for k in ["date", "time", "posted", "published", "fecha"])]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for col in date_cols:
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)  # dayfirst=True for non-US accounts

    # Final info
    content_type = detect_content_type(df)
    print(f"File successfully loaded: {path.name}")
    print(f"   Shape: {df.shape}")
    print(f"   Detected content type: {content_type.upper()}")

    return df


# =============================================================================
# Quick test when running the script directly
# =============================================================================
if __name__ == "__main__":
    try:
        df = load_instagram_csv("InstagramData.csv")
        print("\nFirst 5 rows:")
        print(df.head())
        print("\nData types:")
        print(df.dtypes)
    except Exception as e:
        print(f"Error: {e}")