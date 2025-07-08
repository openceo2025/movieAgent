import streamlit as st
import pandas as pd
import streamlit.components.v1 as components

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

edited_df = st.data_editor(
    data,
    num_rows="dynamic",
    hide_index=True,
    use_container_width=True,
    key="video_editor",
)

if "last_saved_df" not in st.session_state:
    st.session_state["last_saved_df"] = data.copy()

unsaved = not edited_df.equals(st.session_state["last_saved_df"])
st.session_state["unsaved_changes"] = unsaved

if st.button("Save changes", key="save_button"):
    save_data(edited_df, CSV_FILE)
    st.success("Saved to CSV")
    st.session_state["last_saved_df"] = edited_df.copy()
    st.session_state["unsaved_changes"] = False

components.html(
    f"""
    <script>
    (function() {{
        var unsaved = {str(st.session_state["unsaved_changes"]).lower()};
        window.unsavedChanges = unsaved;
        window.addEventListener('beforeunload', function(e) {{
            if (window.unsavedChanges) {{
                var save = confirm('変更が保存されていません。保存しますか?');
                if (save) {{
                    const btns = window.parent.document.querySelectorAll('button');
                    for (const b of btns) {{
                        if (b.innerText.trim() === 'Save changes') {{
                            b.click();
                            break;
                        }}
                    }}
                }}
            }}
        }});
    }})();
    </script>
    """,
    height=0,
)

