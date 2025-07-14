import os
import re
import pandas as pd

# Default generation parameters used when initializing a new CSV
DEFAULT_MODEL = "phi3:mini"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TOP_P = 0.95
DEFAULT_SEED = 1234
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024


def slugify(text: str) -> str:
    """Simplified slugify implementation."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
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


def assign_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame unchanged (ID assignment disabled)."""
    return df


def load_data(path: str) -> pd.DataFrame:
    """Load spreadsheet data from ``path``.

    If the file does not exist an empty DataFrame with default columns is
    returned.
    """
    columns = [
        "selected",
        "id",
        "title",
        "synopsis",
        "llm_model",
        "checkpoint",
        "comfy_vae",
        "comfy_lora",
        "temperature",
        "max_tokens",
        "top_p",
        "seed",
        "batch_count",
        "width",
        "height",
        "story_prompt",
        "bgm_prompt",
        "taste_prompt",
        "character_voice",
        "status",
        "needs_approve",
        "controlnet_image",
    ]
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        df = pd.DataFrame(columns=columns)
        df["selected"] = False
        df["id"] = ""
        df["llm_model"] = DEFAULT_MODEL
        df["checkpoint"] = ""
        df["comfy_vae"] = ""
        df["comfy_lora"] = ""
        df["temperature"] = DEFAULT_TEMPERATURE
        df["max_tokens"] = DEFAULT_MAX_TOKENS
        df["top_p"] = DEFAULT_TOP_P
        df["seed"] = DEFAULT_SEED
        df["batch_count"] = 1
        df["width"] = DEFAULT_WIDTH
        df["height"] = DEFAULT_HEIGHT
        df["controlnet_image"] = ""
    else:
        missing_cols = [c for c in columns if c not in df.columns]
        for c in missing_cols:
            if c == "selected":
                df[c] = False
            elif c == "id":
                df[c] = ""
            else:
                df[c] = ""
        if "llm_model" in missing_cols:
            df["llm_model"] = DEFAULT_MODEL
        if "checkpoint" in missing_cols:
            df["checkpoint"] = ""
        if "comfy_vae" in missing_cols:
            df["comfy_vae"] = ""
        if "comfy_lora" in missing_cols:
            df["comfy_lora"] = ""
        if "temperature" in missing_cols:
            df["temperature"] = DEFAULT_TEMPERATURE
        if "max_tokens" in missing_cols:
            df["max_tokens"] = DEFAULT_MAX_TOKENS
        if "top_p" in missing_cols:
            df["top_p"] = DEFAULT_TOP_P
        if "seed" in missing_cols:
            df["seed"] = DEFAULT_SEED
        if "batch_count" in missing_cols:
            df["batch_count"] = 1
        if "width" in missing_cols:
            df["width"] = DEFAULT_WIDTH
        if "height" in missing_cols:
            df["height"] = DEFAULT_HEIGHT
        if "controlnet_image" in missing_cols:
            df["controlnet_image"] = ""
        df = df[columns]
        df["selected"] = df["selected"].fillna(False).astype(bool)
        df["id"] = df["id"].astype(str)
        df["controlnet_image"] = df["controlnet_image"].fillna("").astype(str)
    return df


def save_data(df: pd.DataFrame, path: str) -> None:
    """Save ``df`` to ``path`` dropping the ``selected`` column."""
    df_copy = df.drop(columns=["selected"], errors="ignore")
    df_copy.to_csv(path, index=False)
