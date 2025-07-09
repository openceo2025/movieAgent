import streamlit as st
import pandas as pd
import subprocess
import requests
import argparse
import sys

# Parse CLI arguments passed after `--` when running via Streamlit
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--debug", action="store_true", help="Enable debug logging")
args, _ = parser.parse_known_args()
DEBUG_MODE = args.debug
if DEBUG_MODE:
    print("[DEBUG] Debug mode enabled")

CSV_FILE = "videos.csv"

# Default generation parameters
DEFAULT_MODEL = "phi3:mini"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TOP_P = 0.95

# ComfyUI API host/port
COMFYUI_HOST = os.getenv("COMFYUI_HOST", "127.0.0.1")
COMFYUI_PORT = os.getenv("COMFYUI_PORT", "8188")


def load_data(path: str) -> pd.DataFrame:
    columns = [
        "selected",
        "id",
        "title",
        "synopsis",
        "llm_model",
        "comfy_model",
        "comfy_vae",
        "comfy_lora",
        "temperature",
        "max_tokens",
        "top_p",
        "story_prompt",
        "bgm_prompt",
        "taste_prompt",
        "character_voice",
        "status",
        "needs_approve",
    ]
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        df = pd.DataFrame(columns=columns)
        df["selected"] = False
        df["id"] = ""
        df["llm_model"] = DEFAULT_MODEL
        df["comfy_model"] = ""
        df["comfy_vae"] = ""
        df["comfy_lora"] = ""
        df["temperature"] = DEFAULT_TEMPERATURE
        df["max_tokens"] = DEFAULT_MAX_TOKENS
        df["top_p"] = DEFAULT_TOP_P
    else:
        missing_cols = [c for c in columns if c not in df.columns]
        for c in missing_cols:
            if c == "selected":
                df[c] = False
            elif c == "id":
                df[c] = [f"{i+1:04d}" for i in range(len(df))]
            else:
                df[c] = ""
        if "llm_model" in missing_cols:
            df["llm_model"] = DEFAULT_MODEL
        if "comfy_model" in missing_cols:
            df["comfy_model"] = ""
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
        df = df[columns]
        df["selected"] = df["selected"].fillna(False).astype(bool)
        df["id"] = df["id"].astype(str)
    return df


def save_data(df: pd.DataFrame, path: str) -> None:
    df_copy = df.drop(columns=["selected"], errors="ignore")
    df_copy.to_csv(path, index=False)


def list_ollama_models() -> list[str]:
    """Return available Ollama models."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
    except Exception:
        return ["phi3:mini"]
    lines = result.stdout.strip().splitlines()
    models = []
    for line in lines[1:]:
        parts = line.split()
        if parts:
            models.append(parts[0])
    return models or ["phi3:mini"]


def list_comfy_models() -> tuple[list[str], list[str], list[str]]:
    """Return (checkpoints, vae, loras) from ComfyUI."""
    base = f"http://{COMFYUI_HOST}:{COMFYUI_PORT}"
    try:
        res = requests.get(f"{base}/models", timeout=5)
        res.raise_for_status()
        folders = res.json()
    except Exception:
        folders = []

    def fetch(folder: str) -> list[str]:
        if folder not in folders:
            return []
        try:
            r = requests.get(f"{base}/models/{folder}", timeout=5)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list):
                return data
        except Exception:
            pass
        return []

    checkpoints = fetch("checkpoints")
    vaes = fetch("vae")
    loras = fetch("loras")
    return checkpoints, vaes, loras


def generate_story_prompt(
    synopsis: str,
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    debug: bool = False,
) -> str:
    prompt = f"Generate a short story based on this synopsis:\n{synopsis}\n"
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "num_predict": max_tokens,
        "top_p": top_p,
        "stream": False,
    }
    if debug:
        print("[DEBUG] Request payload:", payload)
    try:
        res = requests.post(url, json=payload, timeout=60)
        if debug:
            print("[DEBUG] Response status:", res.status_code)
            print("[DEBUG] Raw response:", res.text)
        res.raise_for_status()
        data = res.json()
        if debug:
            print("[DEBUG] Parsed response:", data)
        return data.get("response", "").strip()
    except requests.exceptions.RequestException as e:
        st.error(f"Ollama API error: {e}")
    except ValueError as e:
        st.error(f"Invalid response from Ollama: {e}")
    except Exception as e:
        st.error(f"Error: {e}")
    return None

st.set_page_config(page_title="Video Agent", layout="wide")


def rerun_with_message(message: str) -> None:
    """Trigger st.rerun() and show a message after reload."""
    st.session_state["just_rerun"] = message
    st.rerun()


# Display notice if the page was refreshed by st.rerun()
msg = st.session_state.pop("just_rerun", None)
if msg:
    st.info(msg)

st.title("Streamlit Video Agent")

if "video_df" not in st.session_state:
    st.session_state.video_df = load_data(CSV_FILE)
if "models" not in st.session_state:
    st.session_state.models = list_ollama_models()
if "comfy_models" not in st.session_state:
    (
        st.session_state.comfy_models,
        st.session_state.comfy_vaes,
        st.session_state.comfy_loras,
    ) = list_comfy_models()

df = st.session_state.video_df
for col, options in [
    ("comfy_model", st.session_state.comfy_models),
    ("comfy_vae", st.session_state.comfy_vaes),
    ("comfy_lora", st.session_state.comfy_loras),
]:
    if col not in df.columns:
        df[col] = options[0] if options else ""
    else:
        df[col] = df[col].fillna("")
        if options and (df[col] == "").any():
            df.loc[df[col] == "", col] = options[0]
st.session_state.video_df = df

st.write("### Video Spreadsheet")

edited_df = st.data_editor(
    st.session_state.video_df,
    column_config={
        "selected": st.column_config.CheckboxColumn("Select"),
        "id": st.column_config.TextColumn("ID"),
        "llm_model": st.column_config.SelectboxColumn(
            "Model",
            options=st.session_state.models,
            default="phi3:mini",
        ),
        "comfy_model": st.column_config.SelectboxColumn(
            "Comfy Model",
            options=st.session_state.comfy_models,
            default=st.session_state.comfy_models[0] if st.session_state.comfy_models else "",
        ),
        "comfy_vae": st.column_config.SelectboxColumn(
            "VAE",
            options=st.session_state.comfy_vaes,
            default=st.session_state.comfy_vaes[0] if st.session_state.comfy_vaes else "",
        ),
        "comfy_lora": st.column_config.SelectboxColumn(
            "LoRA",
            options=st.session_state.comfy_loras,
            default=st.session_state.comfy_loras[0] if st.session_state.comfy_loras else "",
        ),
        "temperature": st.column_config.NumberColumn(
            "Temp",
            format="%.2f",
            step=0.05,
            min_value=0.0,
            max_value=1.0,
        ),
        "max_tokens": st.column_config.NumberColumn(
            "Max Tokens",
            min_value=1,
            step=1,
        ),
        "top_p": st.column_config.NumberColumn(
            "Top-p",
            format="%.2f",
            step=0.05,
            min_value=0.0,
            max_value=1.0,
        ),
    },
    num_rows="dynamic",
    hide_index=True,
    use_container_width=True,
    key="video_editor",
)
st.session_state.video_df = edited_df

selected_rows = st.session_state.video_df["selected"] == True
generate_disabled = not selected_rows.any()

if st.button("Generate story prompts", disabled=generate_disabled):
    df = st.session_state.video_df.copy()
    for idx, row in df[selected_rows].iterrows():
        synopsis = row.get("synopsis", "")
        model = row.get("llm_model", DEFAULT_MODEL)
        if pd.isna(model) or model == "":
            model = DEFAULT_MODEL

        temperature = row.get("temperature", DEFAULT_TEMPERATURE)
        if pd.isna(temperature) or temperature == "":
            temperature = DEFAULT_TEMPERATURE
        temperature = float(temperature)

        max_tokens = row.get("max_tokens", DEFAULT_MAX_TOKENS)
        if pd.isna(max_tokens) or max_tokens == "":
            max_tokens = DEFAULT_MAX_TOKENS
        max_tokens = int(max_tokens)

        top_p = row.get("top_p", DEFAULT_TOP_P)
        if pd.isna(top_p) or top_p == "":
            top_p = DEFAULT_TOP_P
        top_p = float(top_p)
        if synopsis:
            prompt = generate_story_prompt(
                synopsis,
                model,
                temperature,
                max_tokens,
                top_p,
                debug=DEBUG_MODE,
            )
            if prompt is not None:
                df.at[idx, "story_prompt"] = prompt
    st.session_state.video_df = df
    save_data(df, CSV_FILE)
    # Refresh the app to show updated prompts immediately
    # Mark that a rerun is triggered so we can notify the user after reload
    rerun_with_message("Page reloaded after generating prompts")

if st.button("Save changes"):
    save_data(st.session_state.video_df, CSV_FILE)
    st.success("Saved to CSV")
