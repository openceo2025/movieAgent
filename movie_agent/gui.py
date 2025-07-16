import argparse
import streamlit as st
import pandas as pd
import os
import random
from pathlib import Path
from .comfyui import (
    list_comfy_models,
    generate_image,
    DEFAULT_CFG,
    DEFAULT_STEPS,
)
from . import framepack
from .ollama import list_ollama_models, generate_story_prompt, DEBUG_MODE
from .csv_manager import (
    load_data,
    save_data,
    slugify,
    unique_path,
    DEFAULT_WIDTH,
    DEFAULT_HEIGHT,
    DEFAULT_FPS,
)

# Parse CLI arguments passed after `--` when launching via Streamlit
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    "--debug",
    action="store_true",
    help="Enable debug logging",
)
args, _ = parser.parse_known_args()
if args.debug:
    print("[DEBUG] Debug mode enabled")

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_FILE = str(BASE_DIR / "videos.csv")

# Default generation parameters
DEFAULT_MODEL = "phi3:mini"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TOP_P = 0.95
DEFAULT_SEED = 1234


st.set_page_config(page_title="Video Agent", layout="wide")


def rerun_with_message(message: str) -> None:
    """Trigger st.rerun() and show a message after reload."""
    st.session_state["just_rerun"] = message
    st.rerun()


def main() -> None:
    """Run the Streamlit GUI."""
    # Display notice if the page was refreshed by st.rerun()
    msg = st.session_state.pop("just_rerun", None)
    if msg:
        st.info(msg)

    st.title("Streamlit Video Agent")

    if "video_df" not in st.session_state:
        st.session_state.video_df = load_data(CSV_FILE)
        st.session_state.last_saved_df = st.session_state.video_df.copy()
    if "last_saved_df" not in st.session_state:
        st.session_state.last_saved_df = st.session_state.video_df.copy()
    if "autosave" not in st.session_state:
        st.session_state.autosave = False
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
        ("checkpoint", st.session_state.comfy_models),
        ("comfy_vae", st.session_state.comfy_vaes),
        ("comfy_lora", st.session_state.comfy_loras),
    ]:
        if col not in df.columns:
            df[col] = ""
        else:
            df[col] = df[col].fillna("")
    st.session_state.video_df = df

    st.write("### Video Spreadsheet")
    df_display = st.session_state.video_df.drop(
        columns=["controlnet_image"], errors="ignore"
    )

    edited_df = st.data_editor(
        df_display,
        column_config={
            "selected": st.column_config.CheckboxColumn("Select"),
            "id": st.column_config.TextColumn("ID"),
            "llm_model": st.column_config.SelectboxColumn(
                "Model",
                options=st.session_state.models,
                default="phi3:mini",
            ),
            "checkpoint": st.column_config.SelectboxColumn(
                "Checkpoint",
                options=st.session_state.comfy_models,
                default=(
                    st.session_state.comfy_models[0]
                    if st.session_state.comfy_models
                    else ""
                ),
            ),
            "comfy_vae": st.column_config.SelectboxColumn(
                "VAE",
                options=st.session_state.comfy_vaes,
                default="",
            ),
            "comfy_lora": st.column_config.SelectboxColumn(
                "LoRA",
                options=st.session_state.comfy_loras,
                default="",
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
            "cfg": st.column_config.NumberColumn(
                "CFG",
                format="%.1f",
                step=0.5,
                min_value=0.0,
            ),
            "steps": st.column_config.NumberColumn(
                "Steps",
                min_value=1,
                step=1,
            ),
            "seed": st.column_config.NumberColumn(
                "Seed",
                step=1,
                format="%d",
            ),
            "batch_count": st.column_config.NumberColumn(
                "Batch",
                min_value=1,
                step=1,
            ),
            "width": st.column_config.NumberColumn(
                "Width",
                min_value=64,
                step=64,
            ),
            "height": st.column_config.NumberColumn(
                "Height",
                min_value=64,
                step=64,
            ),
            "movie_prompt": st.column_config.TextColumn("Movie Prompt"),
            "video_length": st.column_config.NumberColumn(
                "Length (s)",
                min_value=0,
                step=1,
            ),
            "fps": st.column_config.NumberColumn(
                "FPS",
                min_value=1,
                step=1,
            ),
        },
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
        key="video_editor",
    )
    new_df = edited_df.copy()
    if "controlnet_image" in st.session_state.video_df.columns:
        new_df["controlnet_image"] = st.session_state.video_df[
            "controlnet_image"
        ].reindex(new_df.index, fill_value="")
    st.session_state.video_df = new_df[st.session_state.video_df.columns]
    if st.session_state.autosave and not st.session_state.video_df.equals(
        st.session_state.last_saved_df
    ):
        save_data(st.session_state.video_df, CSV_FILE)
        st.session_state.last_saved_df = st.session_state.video_df.copy()
        st.info("Auto-saved to CSV")

    selected_rows = st.session_state.video_df["selected"]
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
        st.session_state.last_saved_df = df.copy()
        # Refresh the app to show updated prompts immediately
        # Mark that a rerun is triggered so we can notify the user after reload
        rerun_with_message("Page reloaded after generating prompts")

    if st.button("Generate images", disabled=generate_disabled):
        df = st.session_state.video_df.copy()
        for idx, row in df[selected_rows].iterrows():
            prompt = row.get("story_prompt", "")
            checkpoint = row.get("checkpoint", "")
            vae = row.get("comfy_vae", "")
            if not prompt:
                st.warning(f"No story prompt for row {row.get('id', idx)}")
                continue
            seed_val = row.get("seed", "")
            if pd.isna(seed_val) or str(seed_val).strip() == "":
                # Empty -> generate a random seed on our side
                seed_val = random.randint(0, 2**32 - 1)
            else:
                seed_val = int(seed_val)

            batch_count = row.get("batch_count", 1)
            if pd.isna(batch_count) or str(batch_count).strip() == "":
                batch_count = 1
            else:
                batch_count = int(batch_count)

            title = row.get("title", "")
            folder = os.path.join(
                BASE_DIR,
                "vids",
                f"{row.get('id', idx)}_{slugify(title)}",
                "panels",
            )

            control_img = row.get("controlnet_image", "")
            width = row.get("width", DEFAULT_WIDTH)
            if pd.isna(width) or str(width).strip() == "":
                width = DEFAULT_WIDTH
            width = int(width)
            height = row.get("height", DEFAULT_HEIGHT)
            if pd.isna(height) or str(height).strip() == "":
                height = DEFAULT_HEIGHT
            height = int(height)

            cfg_val = row.get("cfg", DEFAULT_CFG)
            if pd.isna(cfg_val) or str(cfg_val).strip() == "":
                cfg_val = DEFAULT_CFG
            cfg_val = float(cfg_val)

            steps_val = row.get("steps", DEFAULT_STEPS)
            if pd.isna(steps_val) or str(steps_val).strip() == "":
                steps_val = DEFAULT_STEPS
            steps_val = int(steps_val)

            for b in range(batch_count):
                if seed_val == -1:
                    # -1 is passed through so ComfyUI handles randomization
                    current_seed = -1
                else:
                    current_seed = (
                        seed_val + b if batch_count > 1 else seed_val
                    )
                img_bytes = generate_image(
                    prompt,
                    checkpoint,
                    vae,
                    current_seed,
                    width=width,
                    height=height,
                    cfg=cfg_val,
                    steps=steps_val,
                    control_image=control_img if control_img else None,
                    debug=DEBUG_MODE,
                )
                if img_bytes:
                    os.makedirs(folder, exist_ok=True)
                    out_path = unique_path(os.path.join(folder, "image.png"))
                    with open(out_path, "wb") as f:
                        f.write(img_bytes)
                    st.success(f"Image saved to {out_path}")
                else:
                    message = (
                        "Failed to generate image for row "
                        f"{row.get('id', idx)} (batch {b+1})"
                    )
                    st.error(message)

        st.session_state.video_df = df
        save_data(df, CSV_FILE)
        st.session_state.last_saved_df = df.copy()
        rerun_with_message("Page reloaded after generating images")

    if st.button("Generate videos", disabled=generate_disabled):
        df = st.session_state.video_df.copy()
        for idx, row in df[selected_rows].iterrows():
            title = row.get("title", "")
            base_folder = os.path.join(
                BASE_DIR,
                "vids",
                f"{row.get('id', idx)}_{slugify(title)}",
            )
            panels_dir = os.path.join(base_folder, "panels")
            fps_val = row.get("fps", DEFAULT_FPS)
            if pd.isna(fps_val) or str(fps_val).strip() == "":
                fps_val = DEFAULT_FPS
            fps_val = int(fps_val)
            os.makedirs(base_folder, exist_ok=True)
            out_path = os.path.join(base_folder, "video_raw.mp4")
            result = framepack.generate_video(
                panels_dir,
                fps=fps_val,
                output=out_path,
                debug=DEBUG_MODE,
            )
            if result:
                st.success(f"Video saved to {out_path}")
            else:
                st.error(
                    f"Failed to generate video for row {row.get('id', idx)}"
                )
        st.session_state.video_df = df
        save_data(df, CSV_FILE)
        st.session_state.last_saved_df = df.copy()
        rerun_with_message("Page reloaded after generating videos")

    save_col, auto_col = st.columns([1, 1])
    auto_col.checkbox(
        "Auto-save",
        value=st.session_state.autosave,
        key="autosave",
    )

    if save_col.button("Save CSV"):
        df = st.session_state.video_df.copy()
        st.session_state.video_df = df
        save_data(df, CSV_FILE)
        st.session_state.last_saved_df = df.copy()
        st.success("Saved to CSV")


if __name__ == "__main__":
    main()
