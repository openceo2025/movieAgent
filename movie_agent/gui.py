import argparse
import json
import streamlit as st
import pandas as pd
import os
import random
import logging
from pathlib import Path
from streamlit.components.v1 import html as component_html
from movie_agent.utils import rerun_with_message
from .comfyui import (
    list_comfy_models,
    generate_image,
    DEFAULT_CFG,
    DEFAULT_STEPS,
)
from . import framepack
from .ollama import DEBUG_MODE
from .llm_helpers import select_llm_models, generate_prompt_for_row
from .csv_manager import (
    load_data,
    save_data,
    slugify,
    unique_path,
    DEFAULT_WIDTH,
    DEFAULT_HEIGHT,
    DEFAULT_FPS,
    DEFAULT_VIDEO_LENGTH,
)
from .row_utils import iterate_selected
from movie_agent.logger import logger

# Parse CLI arguments passed after `--` when launching via Streamlit
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    "--debug",
    action="store_true",
    help="Enable debug logging",
)
args, _ = parser.parse_known_args()
if args.debug:
    logger.setLevel(logging.DEBUG)
    logger.debug("Debug mode enabled")

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_FILE = str(BASE_DIR / "videos.csv")

# Default generation parameters
DEFAULT_MODEL = "phi3:mini"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TOP_P = 0.95
DEFAULT_SEED = 31337
DEFAULT_OLLAMA_TIMEOUT = 300

# FramePack default parameters
DEFAULT_LATENT_WINDOW_SIZE = 9
DEFAULT_FP_STEPS = 25
DEFAULT_FP_CFG = 10.0
DEFAULT_GS = 10.0
DEFAULT_RS = 0.0
DEFAULT_GPU_MEMORY_PRESERVATION = 6.0
DEFAULT_USE_TEACACHE = True
DEFAULT_MP4_CRF = 16


st.set_page_config(page_title="Video Agent", layout="wide")


def log_to_console(data: dict) -> None:
    """Send JSON data to the browser console."""
    component_html(
        f"<script>console.log('FramePack request:', {json.dumps(data)});</script>",
        height=0,
    )


def main() -> None:
    """Run the Streamlit GUI."""
    tag_path = BASE_DIR / "tag.json"
    tags: list[str] = []
    if tag_path.exists():
        with tag_path.open(encoding="utf-8") as f:
            data = json.load(f)
            tags = data.get("tags", []) if isinstance(data, dict) else data

    # Display notice if the page was refreshed by st.rerun()
    msg = st.session_state.pop("just_rerun", None)
    if msg:
        st.info(msg)

    st.title("Streamlit Video Agent")
    st.caption(", ".join(tags))

    if "video_df" not in st.session_state:
        st.session_state.video_df = load_data(CSV_FILE)
        st.session_state.last_saved_df = st.session_state.video_df.copy()
    if "last_saved_df" not in st.session_state:
        st.session_state.last_saved_df = st.session_state.video_df.copy()
    if "autosave" not in st.session_state:
        st.session_state.autosave = False
    if "models" not in st.session_state:
        st.session_state.models = select_llm_models(st.session_state.video_df)
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
    if "timeout" not in df.columns:
        df["timeout"] = DEFAULT_OLLAMA_TIMEOUT
    else:
        df["timeout"] = df["timeout"].fillna(DEFAULT_OLLAMA_TIMEOUT)
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
            "timeout": st.column_config.NumberColumn(
                "Timeout",
                min_value=1,
                step=1,
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
    # Ensure the selected column exists and is boolean
    if "selected" not in new_df:
        new_df["selected"] = False
    new_df["selected"] = new_df["selected"].fillna(False).astype(bool)

    if "controlnet_image" in st.session_state.video_df.columns:
        new_df["controlnet_image"] = st.session_state.video_df[
            "controlnet_image"
        ].reindex(new_df.index, fill_value="")

    # Assign sanitized DataFrame back to session state
    st.session_state.video_df = new_df[st.session_state.video_df.columns]
    if st.session_state.autosave and not st.session_state.video_df.equals(
        st.session_state.last_saved_df
    ):
        save_data(st.session_state.video_df, CSV_FILE)
        st.session_state.last_saved_df = st.session_state.video_df.copy()
        st.info("Auto-saved to CSV")

    selected_rows = (
        st.session_state.video_df["selected"].fillna(False).astype(bool)
    )
    generate_disabled = not selected_rows.any()

    if st.button("Generate story prompts", disabled=generate_disabled):
        df = st.session_state.video_df.copy()

        def process(idx: int, row: pd.Series) -> None:
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

            timeout = row.get("timeout", DEFAULT_OLLAMA_TIMEOUT)
            if pd.isna(timeout) or timeout == "":
                timeout = DEFAULT_OLLAMA_TIMEOUT
            timeout = int(timeout)

            nsfw_flag = row.get("nsfw", "")
            if pd.notna(nsfw_flag) and str(nsfw_flag).strip().upper() == "Y":
                synopsis_local = synopsis + " --NSFW"
            else:
                synopsis_local = synopsis

            if synopsis_local:
                prompt = generate_prompt_for_row(
                    row,
                    synopsis_local,
                    model,
                    temperature,
                    max_tokens,
                    top_p,
                    timeout,
                )
                if prompt is not None:
                    df.at[idx, "story_prompt"] = prompt

        iterate_selected(df, process)
        st.session_state.video_df = df
        save_data(df, CSV_FILE)
        st.session_state.last_saved_df = df.copy()
        # Refresh the app to show updated prompts immediately
        # Mark that a rerun is triggered so we can notify the user after reload
        rerun_with_message("Page reloaded after generating prompts")

    if st.button("Generate images", disabled=generate_disabled):
        df = st.session_state.video_df.copy()

        def process(idx: int, row: pd.Series) -> None:
            prompt = row.get("story_prompt", "")
            checkpoint = row.get("checkpoint", "")
            vae = row.get("comfy_vae", "")
            if not prompt:
                st.warning(f"No story prompt for row {row.get('id', idx)}")
                return

            seed_val = row.get("seed", "")
            if pd.isna(seed_val) or str(seed_val).strip() == "":
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
                    current_seed = -1
                else:
                    current_seed = seed_val + b if batch_count > 1 else seed_val
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
                    logger.error(message)
                    st.error(message)

        iterate_selected(df, process)
        st.session_state.video_df = df
        save_data(df, CSV_FILE)
        st.session_state.last_saved_df = df.copy()
        rerun_with_message("Page reloaded after generating images")

    if st.button("Generate videos", disabled=generate_disabled):
        if DEBUG_MODE:
            logger.debug("Generate videos button clicked")
            logger.debug("%d rows selected", int(selected_rows.sum()))
        df = st.session_state.video_df.copy()
        found_any_panels = False

        def process(idx: int, row: pd.Series) -> None:
            nonlocal found_any_panels
            title = row.get("title", "")
            base_folder = os.path.join(
                BASE_DIR,
                "vids",
                f"{row.get('id', idx)}_{slugify(title)}",
            )
            panels_dir = os.path.join(base_folder, "panels")
            if DEBUG_MODE:
                logger.debug(
                    "Processing row %s; panels_dir: %s",
                    row.get('id', idx),
                    panels_dir,
                )

            images = sorted(Path(panels_dir).glob("*.png"))
            if not images:
                if DEBUG_MODE:
                    logger.debug(
                        "No panels found for row %s in %s",
                        row.get('id', idx),
                        panels_dir,
                    )
                st.warning(f"No panels found for row {row.get('id', idx)}")
                return
            found_any_panels = True
            start_image = str(images[0])
            if DEBUG_MODE:
                logger.debug(
                    "Found %d images, start image: %s",
                    len(images),
                    start_image,
                )

            fps_val = row.get("fps", DEFAULT_FPS)
            if pd.isna(fps_val) or str(fps_val).strip() == "":
                fps_val = DEFAULT_FPS
            fps_val = int(fps_val)

            movie_prompt = row.get("movie_prompt", "")
            if pd.isna(movie_prompt):
                movie_prompt = ""

            video_length_val = row.get("video_length", DEFAULT_VIDEO_LENGTH)
            if pd.isna(video_length_val) or str(video_length_val).strip() == "":
                video_length_val = DEFAULT_VIDEO_LENGTH
            video_length_val = float(video_length_val)

            seed_val = row.get("seed", DEFAULT_SEED)
            if pd.isna(seed_val) or str(seed_val).strip() == "":
                seed_val = DEFAULT_SEED
            seed_val = int(seed_val)

            cfg_val = row.get("cfg", DEFAULT_FP_CFG)
            if pd.isna(cfg_val) or str(cfg_val).strip() == "":
                cfg_val = DEFAULT_FP_CFG
            cfg_val = float(cfg_val)

            steps_val = row.get("steps", DEFAULT_FP_STEPS)
            if pd.isna(steps_val) or str(steps_val).strip() == "":
                steps_val = DEFAULT_FP_STEPS
            steps_val = int(steps_val)

            os.makedirs(base_folder, exist_ok=True)
            out_path = os.path.join(base_folder, "video_raw.mp4")

            request_data = {
                "start_image": start_image,
                "movie_prompt": movie_prompt,
                "seed": seed_val,
                "video_length": video_length_val,
                "fps": fps_val,
                "cfg": cfg_val,
                "steps": steps_val,
                "latent_window_size": DEFAULT_LATENT_WINDOW_SIZE,
                "gs": DEFAULT_GS,
                "rs": DEFAULT_RS,
                "gpu_memory_preservation": DEFAULT_GPU_MEMORY_PRESERVATION,
                "use_teacache": DEFAULT_USE_TEACACHE,
                "mp4_crf": DEFAULT_MP4_CRF,
            }
            log_to_console(request_data)
            if DEBUG_MODE:
                logger.debug("request_data: %s", request_data)

            result = framepack.generate_video(
                start_image,
                prompt=movie_prompt,
                seed=seed_val,
                video_length=video_length_val,
                latent_window_size=DEFAULT_LATENT_WINDOW_SIZE,
                steps=steps_val,
                cfg=cfg_val,
                gs=DEFAULT_GS,
                rs=DEFAULT_RS,
                gpu_memory_preservation=DEFAULT_GPU_MEMORY_PRESERVATION,
                use_teacache=DEFAULT_USE_TEACACHE,
                mp4_crf=DEFAULT_MP4_CRF,
                debug=DEBUG_MODE,
            )
            if DEBUG_MODE:
                if result:
                    logger.debug(
                        "framepack.generate_video returned: %s", result
                    )
                else:
                    logger.debug("framepack.generate_video returned None")
            if result:
                st.success(f"Video saved to {out_path}")
            else:
                message = f"Failed to generate video for row {row.get('id', idx)}"
                logger.error(message)
                st.error(message)

        iterate_selected(df, process)
        st.session_state.video_df = df
        save_data(df, CSV_FILE)
        st.session_state.last_saved_df = df.copy()
        if not found_any_panels:
            rerun_with_message("No panels found for selected rows")
        else:
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
