import streamlit as st
import pandas as pd

try:
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
except ModuleNotFoundError:
    AgGrid = GridOptionsBuilder = GridUpdateMode = None

CSV_FILE = "videos.csv"


def load_data(path: str) -> pd.DataFrame:
    columns = [
        "title",
        "synopsis",
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
    else:
        missing_cols = [c for c in columns if c not in df.columns]
        for c in missing_cols:
            df[c] = ""
        df = df[columns]
    return df


def save_data(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)

st.set_page_config(page_title="Video Agent", layout="wide")

st.title("Streamlit Video Agent")

data = load_data(CSV_FILE)

st.write("### Video Spreadsheet")

if AgGrid is None:
    st.error("Required package 'streamlit-aggrid' is not installed.\n"
             "Please run `pip install streamlit-aggrid` and restart the app.")
    st.stop()

gb = GridOptionsBuilder.from_dataframe(data)
gb.configure_default_column(editable=True)
options = gb.build()

grid_return = AgGrid(
    data,
    gridOptions=options,
    update_mode=GridUpdateMode.VALUE_CHANGED,
    fit_columns_on_grid_load=True,
)

updated_df = grid_return["data"]

if st.button("Save changes"):
    save_data(updated_df, CSV_FILE)
    st.success("Saved to CSV")
