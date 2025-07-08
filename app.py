import streamlit as st
import pandas as pd
import subprocess

CSV_FILE = "videos.csv"


def load_data(path: str) -> pd.DataFrame:
    columns = [
        "title",
        "synopsis",
        "llm_model",
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
        df["llm_model"] = "phi3:mini"
    else:
        missing_cols = [c for c in columns if c not in df.columns]
        for c in missing_cols:
            df[c] = ""
        if "llm_model" in missing_cols:
            df["llm_model"] = "phi3:mini"
        df = df[columns]
    return df


def save_data(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)


def list_ollama_models() -> list[str]:
    """Return available Ollama models."""
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, check=True
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


def generate_story_prompt(synopsis: str, model: str) -> str:
    prompt = f"Generate a short story based on this synopsis:\n{synopsis}\n"
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception as e:
        return f"Error: {e}"

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
        "llm_model": st.column_config.SelectboxColumn(
            "Model",
            options=st.session_state.models,
            default="phi3:mini",
        )
    },
    num_rows="dynamic",
    hide_index=True,
    use_container_width=True,
    key="video_editor",
)
st.session_state.video_df = edited_df

if st.button("Generate story prompts"):
    df = st.session_state.video_df.copy()
    for idx, row in df.iterrows():
        synopsis = row.get("synopsis", "")
        model = row.get("llm_model", "phi3:mini")
        if synopsis:
            df.at[idx, "story_prompt"] = generate_story_prompt(synopsis, model)
    st.session_state.video_df = df

if st.button("Save changes"):
    save_data(st.session_state.video_df, CSV_FILE)
    st.success("Saved to CSV")
