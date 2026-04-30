import io
import pandas as pd
from pathlib import Path
from typing import Literal, Optional, Union
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
        source: Union[str, Path, bytes, io.BytesIO],
        min_numeric_ratio: float = 0.6,
) -> pd.DataFrame:
    """
    Loads and cleans Instagram Insights CSV exports in a robust way.

    Accepts:
    - A file path (str or Path)
    - Raw bytes
    - A BytesIO buffer
    - A Streamlit UploadedFile object

    Handles:
    - Headers located in different rows
    - UTF-8 with/without BOM and other encodings
    - Bad lines and duplicate columns
    - Smart numeric and date conversion
    - Automatic content type detection (Posts, Reels, Stories, etc.)
    """

    # ── Normalise source to BytesIO ───────────────────────────────────────
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        buffer = io.BytesIO(path.read_bytes())
        source_name = path.name

    elif isinstance(source, (bytes, bytearray)):
        buffer = io.BytesIO(bytes(source))
        source_name = "uploaded_file.csv"

    elif isinstance(source, io.BytesIO):
        source.seek(0)
        buffer = source
        source_name = "uploaded_file.csv"

    elif hasattr(source, "read"):
        # Streamlit UploadedFile or any file-like object
        raw = source.read()
        if hasattr(source, "seek"):
            source.seek(0)
        buffer = io.BytesIO(raw)
        source_name = getattr(source, "name", "uploaded_file.csv")

    else:
        raise TypeError(f"Unsupported source type: {type(source)}")

    # ── Detect encoding ───────────────────────────────────────────────────
    encoding_used = "utf-8"
    for encoding in ["utf-8-sig", "utf-8", "latin1", "cp1252"]:
        try:
            buffer.seek(0)
            pd.read_csv(buffer, nrows=2, encoding=encoding)
            encoding_used = encoding
            break
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue

    # ── Detect real header row ────────────────────────────────────────────
    buffer.seek(0)
    sample = pd.read_csv(buffer, nrows=10, header=None, encoding=encoding_used)

    header_row_idx = 0
    max_score = 0
    keywords = ["id", "reach", "impressions", "likes", "comments",
                "date", "type", "posted", "published", "saves", "shares"]

    for i in range(min(5, len(sample))):
        row = sample.iloc[i].astype(str)
        score = sum(word in " ".join(row.str.lower().values) for word in keywords)
        if score > max_score:
            max_score = score
            header_row_idx = i

    # ── Load full file ────────────────────────────────────────────────────
    buffer.seek(0)
    df_raw = pd.read_csv(
        buffer,
        skiprows=header_row_idx,
        encoding=encoding_used,
        dtype=str,
        low_memory=False,
        on_bad_lines="skip",
    )

    # ── Clean column names ────────────────────────────────────────────────
    df_raw.columns = (
        df_raw.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^\w]+", "_", regex=True)
        .str.replace(r"^_+|_+$", "", regex=True)
        .str.replace(r"_+", "_", regex=True)
    )

    # ── Deduplicate columns ───────────────────────────────────────────────
    cols = pd.Series(df_raw.columns)
    for dup in df_raw.columns[df_raw.columns.duplicated(keep=False)]:
        count = sum(cols == dup)
        if count > 1:
            indices = cols[cols == dup].index
            for idx, seq in zip(indices[1:], range(1, count)):
                cols.iloc[idx] = f"{dup}_{seq}"
    df_raw.columns = cols

    df = df_raw.copy()

    # ── Protect ID columns ────────────────────────────────────────────────
    id_cols = [c for c in df.columns if "id" in c.lower()]
    for col in id_cols:
        df[col] = df[col].astype(str).str.replace(r"\.0$", "", regex=True)

    # ── Safe numeric conversion ───────────────────────────────────────────
    for col in df.columns:
        if col in id_cols:
            continue

        serie = df[col].astype(str).str.strip()

        if serie.str.contains(r"[%,$kKmM]", regex=True).any():
            serie = serie.str.replace(r"[,%$]", "", regex=True)
            serie = serie.str.replace(
                r"^(\d+\.?\d*)k$",
                lambda m: str(float(m.group(1)) * 1000), regex=True
            )
            serie = serie.str.replace(
                r"^(\d+\.?\d*)m$",
                lambda m: str(float(m.group(1)) * 1_000_000), regex=True
            )

        numeric = pd.to_numeric(serie, errors="coerce")
        if numeric.notna().mean() >= min_numeric_ratio:
            df[col] = numeric

    # ── Date conversion ───────────────────────────────────────────────────
    date_keywords = ["date", "time", "posted", "published", "fecha", "publish_time"]
    date_cols = [c for c in df.columns if any(k in c.lower() for k in date_keywords)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for col in date_cols:
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)

    # ── Drop fully empty rows ─────────────────────────────────────────────
    df.dropna(how="all", inplace=True)
    df.reset_index(drop=True, inplace=True)

    content_type = detect_content_type(df)
    print(f"✅ Loaded: {source_name}  |  {df.shape[0]} rows × {df.shape[1]} cols  |  {content_type.upper()}")

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
