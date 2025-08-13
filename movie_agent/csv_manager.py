import os
import re
import pandas as pd

from .csv_schema import (
    DEFAULT_CFG,
    DEFAULT_STEPS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TOP_P,
    DEFAULT_SEED,
    DEFAULT_WIDTH,
    DEFAULT_HEIGHT,
    DEFAULT_FPS,
    DEFAULT_VIDEO_LENGTH,
    DEFAULT_TIMEOUT,
    VIDEO_COLUMNS,
    VIDEO_DEFAULTS,
    IMAGE_COLUMNS,
    IMAGE_DEFAULTS,
)


def slugify(text: str) -> str:
    """Simplified slugify implementation."""
    text = text.lower().strip()
    text = re.sub(r"[^\w.-]+", "_", text, flags=re.UNICODE)
    return text.strip("_") or "item"


def unique_path(path: str) -> str:
    """Return a non-existing filepath by adding numeric suffixes."""
    base, ext = os.path.splitext(path)
    candidate = path
    counter = 1
    while os.path.exists(candidate):
        candidate = f"{base}_{counter}{ext}"
        counter += 1
    return candidate


def load_data(path: str) -> pd.DataFrame:
    """Load spreadsheet data from ``path``.

    If the file does not exist an empty DataFrame with default columns is
    returned.
    """
    columns = VIDEO_COLUMNS
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        df = pd.DataFrame(columns=columns)
        for col, default in VIDEO_DEFAULTS.items():
            df[col] = default
    else:
        missing_cols = [c for c in columns if c not in df.columns]
        for c in missing_cols:
            df[c] = VIDEO_DEFAULTS.get(c, "")
        df = df[columns]
        df["selected"] = df["selected"].fillna(False).astype(bool)
        df["id"] = df["id"].astype(str)
        df["controlnet_image"] = df["controlnet_image"].fillna("").astype(str)
        df["movie_prompt"] = df["movie_prompt"].fillna("").astype(str)
        df["video_length"] = pd.to_numeric(df["video_length"], errors="coerce").fillna(DEFAULT_VIDEO_LENGTH).astype(int)
        df["fps"] = pd.to_numeric(df["fps"], errors="coerce").fillna(DEFAULT_FPS).astype(int)
    return df


def save_data(df: pd.DataFrame, path: str) -> None:
    """Save ``df`` to ``path`` dropping the ``selected`` column."""
    df_copy = df.drop(columns=["selected"], errors="ignore")
    df_copy.to_csv(path, index=False)


def load_image_data(path: str) -> pd.DataFrame:
    """Load image spreadsheet data from ``path``.

    Creates a DataFrame with the expected columns when the file does not
    exist. Missing columns are added with default values.
    """
    columns = IMAGE_COLUMNS

    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        df = pd.DataFrame(columns=columns)
        for col, default in IMAGE_DEFAULTS.items():
            df[col] = default
    else:
        missing_cols = [c for c in columns if c not in df.columns]
        for c in missing_cols:
            df[c] = IMAGE_DEFAULTS.get(c, "")

        df = df[columns]
        df["selected"] = df["selected"].fillna(False).astype(bool)
        df["nsfw"] = df["nsfw"].fillna(False).astype(bool)
        df["id"] = df["id"].astype(str)
        df["category"] = df["category"].fillna("").astype(str)
        df["tags"] = df["tags"].fillna("").astype(str)
        df["ja_prompt"] = df["ja_prompt"].fillna("").astype(str)
        df["llm_model"] = df["llm_model"].fillna(DEFAULT_MODEL).astype(str)
        df["llm_environment"] = df["llm_environment"].fillna(IMAGE_DEFAULTS["llm_environment"]).astype(str)
        df["image_prompt"] = df["image_prompt"].fillna("").astype(str)
        df["negative_prompt"] = df["negative_prompt"].fillna("").astype(str)
        df["sfw_negative_prompt"] = df["sfw_negative_prompt"].fillna("").astype(str)
        df["image_path"] = df["image_path"].fillna("").astype(str)
        df["post_url"] = df["post_url"].fillna("").astype(str)
        df["post_site"] = df["post_site"].fillna("").astype(str)
        df["post_id"] = df["post_id"].fillna("").astype(str)
        df["wordpress_site"] = df["wordpress_site"].fillna("").astype(str)
        df["wordpress_account"] = df["wordpress_account"].fillna("").astype(str)
        df["timeout"] = pd.to_numeric(df["timeout"], errors="coerce").fillna(DEFAULT_TIMEOUT).astype(int)
        for vcol in ["views_yesterday", "views_week", "views_month"]:
            df[vcol] = pd.to_numeric(df[vcol], errors="coerce").fillna(0).astype(int)
    return df

