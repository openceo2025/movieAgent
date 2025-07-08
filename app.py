import streamlit as st
import pandas as pd
import subprocess
import requests

CSV_FILE = "videos.csv"

# Default generation parameters
DEFAULT_MODEL = "phi3:mini"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TOP_P = 0.95


def load_data(path: str) -> pd.DataFrame:
    columns = [
        "selected",
        "title",
        "synopsis",
        "llm_model",
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
        df["llm_model"] = DEFAULT_MODEL
        df["temperature"] = DEFAULT_TEMPERATURE
        df["max_tokens"] = DEFAULT_MAX_TOKENS
        df["top_p"] = DEFAULT_TOP_P
    else:
        missing_cols = [c for c in columns if c not in df.columns]
        for c in missing_cols:
            if c == "selected":
                df[c] = False
            else:
                df[c] = ""
        if "llm_model" in missing_cols:
            df["llm_model"] = DEFAULT_MODEL
        if "temperature" in missing_cols:
            df["temperature"] = DEFAULT_TEMPERATURE
        if "max_tokens" in missing_cols:
            df["max_tokens"] = DEFAULT_MAX_TOKENS
        if "top_p" in missing_cols:
            df["top_p"] = DEFAULT_TOP_P
        df = df[columns]
        df["selected"] = df["selected"].fillna(False).astype(bool)
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


def generate_story_prompt(
    synopsis: str,
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
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
    try:
        res = requests.post(url, json=payload, timeout=60)
        res.raise_for_status()
        data = res.json()
        return data.get("response", "").strip()
    except requests.exceptions.RequestException as e:
        st.error(f"Ollama API error: {e}")
    except ValueError as e:
        st.error(f"Invalid response from Ollama: {e}")
    except Exception as e:
        st.error(f"Error: {e}")
    return None

st.set_page_config(page_title="Video Agent", layout="wide")

st.title("Streamlit Video Agent")

if "video_df" not in st.session_state:
    st.session_state.video_df = load_data(CSV_FILE)
if "models" not in st.session_state:
    st.session_state.models = list_ollama_models()

st.write("### Video Spreadsheet")

edited_df = st.data_editor(
    st.session_state.video_df,
    column_config={
        "selected": st.column_config.CheckboxColumn("Select"),
        "llm_model": st.column_config.SelectboxColumn(
            "Model",
            options=st.session_state.models,
            default="phi3:mini",
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
                synopsis, model, temperature, max_tokens, top_p
            )
            if prompt is not None:
                df.at[idx, "story_prompt"] = prompt
    st.session_state.video_df = df
    save_data(df, CSV_FILE)
    # Refresh the app to show updated prompts immediately
    st.rerun()

if st.button("Save changes"):
    save_data(st.session_state.video_df, CSV_FILE)
    st.success("Saved to CSV")
