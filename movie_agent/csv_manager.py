import os
import re
import pandas as pd

from .comfyui import DEFAULT_CFG, DEFAULT_STEPS

# Default generation parameters used when initializing a new CSV
DEFAULT_MODEL = "phi3:mini"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TOP_P = 0.95
DEFAULT_SEED = 31337
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
DEFAULT_FPS = 24
DEFAULT_VIDEO_LENGTH = 3


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
        "cfg",
        "steps",
        "seed",
        "batch_count",
        "width",
        "height",
        "story_prompt",
        "bgm_prompt",
        "taste_prompt",
        "character_voice",
        "movie_prompt",
        "video_length",
        "fps",
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
        df["cfg"] = DEFAULT_CFG
        df["steps"] = DEFAULT_STEPS
        df["seed"] = DEFAULT_SEED
        df["batch_count"] = 1
        df["width"] = DEFAULT_WIDTH
        df["height"] = DEFAULT_HEIGHT
        df["movie_prompt"] = ""
        df["video_length"] = DEFAULT_VIDEO_LENGTH
        df["fps"] = DEFAULT_FPS
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
        if "cfg" in missing_cols:
            df["cfg"] = DEFAULT_CFG
        if "steps" in missing_cols:
            df["steps"] = DEFAULT_STEPS
        if "seed" in missing_cols:
            df["seed"] = DEFAULT_SEED
        if "batch_count" in missing_cols:
            df["batch_count"] = 1
        if "width" in missing_cols:
            df["width"] = DEFAULT_WIDTH
        if "height" in missing_cols:
            df["height"] = DEFAULT_HEIGHT
        if "movie_prompt" in missing_cols:
            df["movie_prompt"] = ""
        if "video_length" in missing_cols:
            df["video_length"] = DEFAULT_VIDEO_LENGTH
        if "fps" in missing_cols:
            df["fps"] = DEFAULT_FPS
        if "controlnet_image" in missing_cols:
            df["controlnet_image"] = ""
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
