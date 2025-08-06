import argparse
import os
from datetime import datetime
from pathlib import Path

import requests
import streamlit as st
from movie_agent.comfyui import (
    list_comfy_models,
    generate_image,
    DEFAULT_CFG,
    DEFAULT_STEPS,
)
from movie_agent.ollama import (
    list_ollama_models,
    generate_story_prompt,
    DEBUG_MODE,
)
from movie_agent.csv_manager import (
    load_image_data,
    save_data,
    slugify,
    unique_path,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TOP_P,
    DEFAULT_WIDTH,
    DEFAULT_HEIGHT,
    DEFAULT_SEED,
)

# Parse CLI arguments passed after `--` when launching via Streamlit
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--debug", action="store_true", help="Enable debug logging")
args, _ = parser.parse_known_args()
if args.debug:
    print("[DEBUG] Debug mode enabled")

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_FILE = str(BASE_DIR / "images.csv")
AUTOPOSTER_API_URL = os.getenv("AUTOPOSTER_API_URL", "http://127.0.0.1:9000")

st.set_page_config(page_title="Image Agent", layout="wide")


def main() -> None:
    """Run the Streamlit UI for image generation and posting."""
    msg = st.session_state.pop("just_rerun", None)
    if msg:
        st.info(msg)

    st.title("Image Generation Agent")

    if "image_df" not in st.session_state:
        st.session_state.image_df = load_image_data(CSV_FILE)
        st.session_state.last_saved_df = st.session_state.image_df.copy()
    if "last_saved_df" not in st.session_state:
        st.session_state.last_saved_df = st.session_state.image_df.copy()
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

    df = st.session_state.image_df
    if "llm_model" not in df.columns:
        idx = df.columns.get_loc("image_prompt") if "image_prompt" in df.columns else len(df.columns)
        df.insert(idx, "llm_model", DEFAULT_MODEL)
    else:
        df["llm_model"] = df["llm_model"].fillna(DEFAULT_MODEL)
    for col in ["checkpoint", "comfy_vae"]:
        if col not in df.columns:
            df[col] = ""
        else:
            df[col] = df[col].fillna("")
    st.session_state.image_df = df

    st.write("### Image Spreadsheet")
    edited_df = st.data_editor(
        df,
        column_config={
            "selected": st.column_config.CheckboxColumn("Select"),
            "id": st.column_config.TextColumn("ID"),
            "category": st.column_config.TextColumn("Category"),
            "tags": st.column_config.TextColumn("Tags"),
            "nsfw": st.column_config.CheckboxColumn("NSFW"),
            "ja_prompt": st.column_config.TextColumn("Japanese Prompt"),
            "llm_model": st.column_config.SelectboxColumn(
                "LLM Model", options=st.session_state.models
            ),
            "image_prompt": st.column_config.TextColumn("Image Prompt"),
            "image_path": st.column_config.TextColumn("Image Path"),
            "post_url": st.column_config.TextColumn("Post URL"),
            "views_yesterday": st.column_config.NumberColumn(
                "Views Yesterday", min_value=0
            ),
            "views_week": st.column_config.NumberColumn("Views Week", min_value=0),
            "views_month": st.column_config.NumberColumn("Views Month", min_value=0),
            "checkpoint": st.column_config.SelectboxColumn(
                "Checkpoint", options=st.session_state.comfy_models
            ),
            "comfy_vae": st.column_config.SelectboxColumn(
                "VAE", options=st.session_state.comfy_vaes
            ),
            "steps": st.column_config.NumberColumn("Steps", min_value=1),
            "seed": st.column_config.NumberColumn("Seed", min_value=0),
            "batch_count": st.column_config.NumberColumn(
                "Batch", min_value=1, max_value=10
            ),
            "width": st.column_config.NumberColumn("Width", min_value=64),
            "height": st.column_config.NumberColumn("Height", min_value=64),
        },
        num_rows="dynamic",
        hide_index=True,
    )
    st.session_state.image_df = edited_df

    prompt_col, gen_col, post_col, anal_col = st.columns(4)

    if prompt_col.button("Generate prompt"):
        df = st.session_state.image_df
        selected = df[df["selected"]]
        for idx, row in selected.iterrows():
            ja = row.get("ja_prompt", "")
            if not row.get("image_prompt"):
                if ja:
                    try:
                        model = row.get("llm_model")
                        if not model:
                            model = DEFAULT_MODEL
                        prompt = generate_story_prompt(
                            ja,
                            model=model,
                            temperature=row.get("temperature", DEFAULT_TEMPERATURE),
                            max_tokens=row.get("max_tokens", DEFAULT_MAX_TOKENS),
                            top_p=row.get("top_p", DEFAULT_TOP_P),
                        )
                        df.at[idx, "image_prompt"] = prompt
                    except Exception as e:
                        st.error(
                            f"Prompt generation failed for row {row.get('id', idx)}: {e}"
                        )
            else:
                st.warning(
                    f"Row {row.get('id', idx)} already has an image_prompt"
                )
        st.session_state.image_df = df
        if st.session_state.autosave:
            save_data(df, CSV_FILE)

    if gen_col.button("Generate images"):
        df = st.session_state.image_df
        selected = df[df["selected"]]
        for idx, row in selected.iterrows():
            prompt = row.get("image_prompt", "")
            if not prompt:
                st.warning(f"No image_prompt for row {row.get('id', idx)}")
                continue
            checkpoint = row.get("checkpoint") or ""
            vae = row.get("comfy_vae") or ""
            seed_val = int(row.get("seed", DEFAULT_SEED) or DEFAULT_SEED)
            steps_val = int(row.get("steps", DEFAULT_STEPS) or DEFAULT_STEPS)
            width_val = int(row.get("width", DEFAULT_WIDTH) or DEFAULT_WIDTH)
            height_val = int(row.get("height", DEFAULT_HEIGHT) or DEFAULT_HEIGHT)
            batch = int(row.get("batch_count", 1) or 1)

            for b in range(batch):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                category = slugify(row.get("category", ""))
                model_slug = slugify(checkpoint or "model")
                filename = f"{timestamp}_{model_slug}_{b}.png"
                base_dir = Path(BASE_DIR) / "imgs" / category
                os.makedirs(base_dir, exist_ok=True)
                out_path = unique_path(str(base_dir / filename))

                try:
                    img_bytes = generate_image(
                        prompt,
                        checkpoint=checkpoint,
                        vae=vae,
                        seed=seed_val + b,
                        width=width_val,
                        height=height_val,
                        cfg=row.get("cfg", DEFAULT_CFG),
                        steps=steps_val,
                        debug=DEBUG_MODE,
                    )
                    if img_bytes:
                        with open(out_path, "wb") as f:
                            f.write(img_bytes)
                        df.at[idx, "image_path"] = out_path
                        st.success(f"Saved image to {out_path}")
                    else:
                        st.error(
                            f"Failed to generate image for row {row.get('id', idx)}"
                        )
                except Exception as e:
                    st.error(
                        f"Image generation error for row {row.get('id', idx)}: {e}"
                    )
        st.session_state.image_df = df
        if st.session_state.autosave:
            save_data(df, CSV_FILE)

    if post_col.button("Post"):
        df = st.session_state.image_df
        selected = df[df["selected"]]
        for idx, row in selected.iterrows():
            image_path = row.get("image_path", "")
            if not image_path or not os.path.exists(image_path):
                st.warning(f"No image file for row {row.get('id', idx)}")
                continue
            title = row.get("id", "")
            tags = row.get("tags", "")
            try:
                files = {"file": open(image_path, "rb")}
            except Exception:
                st.error(f"Unable to open image {image_path}")
                continue
            try:
                res = requests.post(
                    f"{AUTOPOSTER_API_URL}/post",
                    data={"title": title, "tags": tags},
                    files=files,
                    timeout=10,
                )
                res.raise_for_status()
                url = res.json().get("url", "")
                if url:
                    df.at[idx, "post_url"] = url
                    st.success(f"Posted: {url}")
                else:
                    st.error("No URL returned from autoPoster")
            except Exception as e:
                st.error(
                    f"Posting failed for row {row.get('id', idx)}: {e}"
                )
            finally:
                files["file"].close()
        st.session_state.image_df = df
        if st.session_state.autosave:
            save_data(df, CSV_FILE)

    if anal_col.button("Analysis"):
        df = st.session_state.image_df
        selected = df[df["selected"]]
        for idx, row in selected.iterrows():
            url = row.get("post_url", "")
            if not url:
                st.warning(f"No post_url for row {row.get('id', idx)}")
                continue
            try:
                res = requests.get(
                    f"{AUTOPOSTER_API_URL}/stats",
                    params={"url": url},
                    timeout=10,
                )
                res.raise_for_status()
                data = res.json()
                df.at[idx, "views_yesterday"] = data.get(
                    "views_yesterday", row.get("views_yesterday", 0)
                )
                df.at[idx, "views_week"] = data.get(
                    "views_week", row.get("views_week", 0)
                )
                df.at[idx, "views_month"] = data.get(
                    "views_month", row.get("views_month", 0)
                )
            except Exception as e:
                st.error(f"Analysis failed for row {row.get('id', idx)}: {e}")
        st.session_state.image_df = df
        if st.session_state.autosave:
            save_data(df, CSV_FILE)

    save_col, auto_col = st.columns([1, 1])
    auto_col.checkbox("Auto-save", value=st.session_state.autosave, key="autosave")
    if save_col.button("Save CSV"):
        df = st.session_state.image_df.copy()
        st.session_state.image_df = df
        save_data(df, CSV_FILE)
        st.session_state.last_saved_df = df.copy()
        st.success("Saved to CSV")


if __name__ == "__main__":
    main()
