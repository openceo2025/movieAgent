import argparse
import os
from datetime import datetime
from pathlib import Path

import base64
import json
import random
from typing import Optional, Dict, Any

import requests
import streamlit as st
import pandas as pd
from movie_agent.comfyui import (
    list_comfy_models,
    generate_image,
    DEFAULT_CFG,
    DEFAULT_STEPS,
    DEFAULT_NEGATIVE_PROMPT,
)
from movie_agent.ollama import (
    list_ollama_models,
    generate_story_prompt,
    DEBUG_MODE,
)
from movie_agent.lmstudio import generate_story_prompt_lmstudio, list_lmstudio_models
from movie_agent.csv_manager import (
    load_image_data,
    save_data,
    slugify,
    DEFAULT_MODEL,
    DEFAULT_WIDTH,
    DEFAULT_HEIGHT,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
)
from movie_agent.row_utils import iterate_selected

# Parse CLI arguments passed after `--` when launching via Streamlit
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--debug", action="store_true", help="Enable debug logging")
args, _ = parser.parse_known_args()
if args.debug:
    print("[DEBUG] Debug mode enabled")

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_FILE = str(BASE_DIR / "images.csv")
AUTOPOSTER_API_URL = os.getenv("AUTOPOSTER_API_URL", "http://127.0.0.1:9000")
WORDPRESS_API_URL = os.getenv(
    "WORDPRESS_API_URL", "http://localhost:8765/wordpress/post"
)
DEFAULT_TIMEOUT = 300
DEFAULT_BATCH = 1

st.set_page_config(page_title="Image Agent", layout="wide")


def coerce_int(value, default, label, row_id, min_value=None):
    """Safely convert ``value`` to int or return ``default`` with a toast."""
    if pd.isna(value) or str(value).strip() == "":
        return default
    try:
        value = int(value)
    except (ValueError, TypeError):
        st.toast(f"Invalid {label} for row {row_id}; using default {default}")
        return default
    if min_value is not None and value < min_value:
        st.toast(
            f"{label.capitalize()} must be >= {min_value} for row {row_id}; using {default}"
        )
        return default
    return value


def coerce_float(value, default, label, row_id, min_value=None):
    """Safely convert ``value`` to float or return ``default`` with a toast."""
    if pd.isna(value) or str(value).strip() == "":
        return default
    try:
        value = float(value)
    except (ValueError, TypeError):
        st.toast(f"Invalid {label} for row {row_id}; using default {default}")
        return default
    if min_value is not None and value < min_value:
        st.toast(
            f"{label.capitalize()} must be >= {min_value} for row {row_id}; using {default}"
        )
        return default
    return value


def rerun_with_message(message: str) -> None:
    """Trigger st.rerun() and show a message after reload."""
    st.session_state["just_rerun"] = message
    st.rerun()


def build_image_prompt_context(row: pd.Series) -> str:
    """Build the context string for image prompt generation."""
    base = row.get("ja_prompt", "")
    nsfw = bool(row.get("nsfw"))
    language = str(row.get("id") or "").strip()
    synopsis = (
        "Create an English image-generation prompt.\n"
        f"Category: {str(row.get('category') or '')}\n"
        f"Tags: {str(row.get('tags') or '')}\n"
        f"Base prompt (Japanese): {base}\n"
        f"NSFW allowed: {nsfw}\n"
        "Return only the final English image-generation prompt.\n"
        "Do not include any explanations, notes, or reasoning text.\n"
    )
    if pd.notna(language) and language != "":
        synopsis += (
            f"Language hint: {language}\n"
            "Ensure the prompt depicts people whose appearance reflects typical traits of regions where this language is primarily spoken."
        )
    return synopsis


def post_to_wordpress(row: pd.Series) -> Optional[Dict[str, Any]]:
    """Post image metadata and files to a WordPress server.

    Returns a dict with ``link``, ``site``, and ``id`` keys on success.
    """

    tags_list = [t.strip() for t in row.get("tags", "").split(",") if t.strip()]
    first_tag = tags_list[0] if tags_list else ""
    title_parts = ["AI image", row.get("category", ""), first_tag]
    title = " ".join([p for p in title_parts if p])
    content = ", ".join(tags_list)
    image_dir = Path(row.get("image_path", ""))
    if not image_dir.exists():
        raise FileNotFoundError(f"Image path not found: {image_dir}")

    # The `wordpress_site` field is a key (not a URL) used for account/site lookup.
    site = (row.get("wordpress_site") or os.getenv("WORDPRESS_SITE", "")).strip()
    if not site:
        st.error("WordPressサイトが指定されていません")
        return None

    media = []
    for p in sorted(image_dir.glob("*")):
        if p.is_file():
            with open(p, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
            media.append({"filename": p.name, "data": encoded})

    account = row.get("wordpress_account")
    if account is None or str(account).strip() == "":
        st.error("WordPressアカウントが指定されていません")
        return None
    payload = {
        "account": account,
        "title": title,
        "content": content,
        "media": media,
        "categories": [row["category"]],
        "tags": tags_list,
    }

    api_url = WORDPRESS_API_URL
    payload["site"] = site  # site key (not a full URL)

    debug_payload = payload.copy()
    debug_payload["media"] = [m["filename"] for m in media]
    print(
        f"[DEBUG] POST {api_url} payload: {json.dumps(debug_payload, ensure_ascii=False)}"
    )

    try:
        resp = requests.post(api_url, json=payload, timeout=10)
        resp.raise_for_status()
    except requests.HTTPError as e:
        st.error(f"WordPress投稿に失敗しました: {e}")
        return None
    except requests.RequestException as e:
        st.error(f"WordPress投稿に失敗しました: {e}")
        return None
    if resp.status_code not in (200, 201):
        st.error(
            f"WordPress投稿に失敗しました: {resp.status_code} {resp.text}"
        )
        return None
    data = resp.json()
    link = data.get("link")
    site_resp = data.get("site")
    post_id = data.get("id")
    if link and site_resp is not None and post_id is not None:
        return {"link": link, "site": site_resp, "id": post_id}
    st.warning("WordPressから投稿URLが返されませんでした")
    return None


def fetch_view_counts(site: str, post_id: str) -> Dict[str, int]:
    """Fetch view counts for a WordPress post via autoPoster.

    Returns a dict with ``views_yesterday``, ``views_week``, and
    ``views_month`` keys.
    """

    results: Dict[str, int] = {}
    for days, column in [
        (1, "views_yesterday"),
        (7, "views_week"),
        (30, "views_month"),
    ]:
        res = requests.get(
            f"{AUTOPOSTER_API_URL}/wordpress/stats/views",
            params={"site": site, "post_id": post_id, "days": days},
            timeout=10,
        )
        res.raise_for_status()
        data = res.json()
        views = data.get("views", [])
        results[column] = views[0] if views else 0
    return results


def select_llm_models(df: pd.DataFrame) -> list[str]:
    """Return available LLM models based on ``llm_environment`` column."""
    if (
        "llm_environment" in df.columns
        and df["llm_environment"].astype(str).str.lower().eq("lmstudio").any()
    ):
        return list_lmstudio_models()
    return list_ollama_models()


def generate_prompt_for_row(
    row: pd.Series,
    context: str,
    model: str,
    temperature: float,
    max_tokens: Optional[int],
    top_p: Optional[float],
    timeout: int,
) -> Optional[str]:
    """Generate a prompt using the appropriate LLM backend."""
    env = str(row.get("llm_environment", "")).strip().lower()
    if env == "lmstudio":
        return generate_story_prompt_lmstudio(
            context,
            model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            timeout=timeout,
        )
    return generate_story_prompt(
        context,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        debug=DEBUG_MODE,
        timeout=timeout,
    )


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
        st.session_state.models = select_llm_models(st.session_state.image_df)
    if "comfy_models" not in st.session_state:
        (
            st.session_state.comfy_models,
            st.session_state.comfy_vaes,
            st.session_state.comfy_loras,
        ) = list_comfy_models()

    default_checkpoint = (
        st.session_state.comfy_models[0]
        if st.session_state.comfy_models
        else ""
    )

    df = st.session_state.image_df
    if "llm_model" not in df.columns:
        idx = df.columns.get_loc("image_prompt") if "image_prompt" in df.columns else len(df.columns)
        df.insert(idx, "llm_model", DEFAULT_MODEL)
    else:
        df["llm_model"] = df["llm_model"].fillna(DEFAULT_MODEL)
    if "llm_environment" not in df.columns:
        idx = df.columns.get_loc("image_prompt") if "image_prompt" in df.columns else len(df.columns)
        df.insert(idx, "llm_environment", "Ollama")
    else:
        df["llm_environment"] = df["llm_environment"].fillna("Ollama")

    # Ensure posting-related columns exist
    if "post_url" not in df.columns:
        idx = df.columns.get_loc("image_path") + 1 if "image_path" in df.columns else len(df.columns)
        df.insert(idx, "post_url", "")
    else:
        df["post_url"] = df["post_url"].fillna("")

    if "post_site" not in df.columns:
        idx = df.columns.get_loc("post_url") + 1
        df.insert(idx, "post_site", "")
    else:
        df["post_site"] = df["post_site"].fillna("")

    if "post_id" not in df.columns:
        idx = df.columns.get_loc("post_site") + 1
        df.insert(idx, "post_id", "")
    else:
        df["post_id"] = df["post_id"].fillna("")

    if "wordpress_site" not in df.columns:
        idx = df.columns.get_loc("post_id") + 1
        df.insert(idx, "wordpress_site", "")
    else:
        df["wordpress_site"] = df["wordpress_site"].fillna("")
    if "wordpress_account" not in df.columns:
        idx = df.columns.get_loc("wordpress_site") + 1
        df.insert(idx, "wordpress_account", "")
    else:
        df["wordpress_account"] = df["wordpress_account"].fillna("")
    # "wordpress_site" values are keys, not full URLs.
    for col in ["checkpoint", "comfy_vae"]:
        if col not in df.columns:
            if col == "checkpoint":
                df[col] = default_checkpoint
            else:
                df[col] = ""
        else:
            if col == "checkpoint":
                df[col] = df[col].replace("", default_checkpoint).fillna(
                    default_checkpoint
                )
            else:
                df[col] = df[col].fillna("")
    if "timeout" not in df.columns:
        df["timeout"] = DEFAULT_TIMEOUT
    else:
        df["timeout"] = df["timeout"].fillna(DEFAULT_TIMEOUT)
    if "cfg" not in df.columns:
        df["cfg"] = DEFAULT_CFG
    else:
        df["cfg"] = df["cfg"].replace("", DEFAULT_CFG).fillna(DEFAULT_CFG)
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
            "llm_environment": st.column_config.SelectboxColumn(
                "LLM Environment", options=["Ollama", "LMStudio"]
            ),
            "image_prompt": st.column_config.TextColumn("Image Prompt"),
            "negative_prompt": st.column_config.TextColumn("Negative Prompt"),
            "sfw_negative_prompt": st.column_config.TextColumn("SFW Negative Prompt"),
            "image_path": st.column_config.LinkColumn("Image Path"),
            "post_url": st.column_config.TextColumn("Post URL"),
            "post_site": st.column_config.TextColumn("Post Site"),
            "post_id": st.column_config.TextColumn("Post ID"),
            "views_yesterday": st.column_config.NumberColumn(
                "Views Yesterday", min_value=0
            ),
            "views_week": st.column_config.NumberColumn("Views Week", min_value=0),
            "views_month": st.column_config.NumberColumn("Views Month", min_value=0),
            "checkpoint": st.column_config.SelectboxColumn(
                "Checkpoint",
                options=st.session_state.comfy_models,
                default=default_checkpoint,
            ),
            "comfy_vae": st.column_config.SelectboxColumn(
                "VAE", options=st.session_state.comfy_vaes
            ),
            "steps": st.column_config.NumberColumn("Steps", min_value=1),
            "seed": st.column_config.NumberColumn("Seed", min_value=-1),
            "batch_count": st.column_config.NumberColumn(
                "Batch", min_value=1, max_value=10
            ),
            "width": st.column_config.NumberColumn("Width", min_value=64),
            "height": st.column_config.NumberColumn("Height", min_value=64),
            "cfg": st.column_config.NumberColumn("CFG", min_value=1.0),
            "timeout": st.column_config.NumberColumn("Timeout", min_value=1),
        },
        num_rows="dynamic",
        hide_index=True,
    )
    st.session_state.image_df = edited_df
    if len(edited_df) > len(st.session_state.last_saved_df):
        save_data(edited_df, CSV_FILE)
        st.session_state.last_saved_df = edited_df.copy()

    prompt_col, gen_col, post_col, anal_col = st.columns(4)


    if prompt_col.button("Generate prompt"):
        df = st.session_state.image_df.copy()

        def process(idx: int, row: pd.Series) -> None:
            base = row.get("ja_prompt", "")
            nsfw = bool(row.get("nsfw"))
            if not row.get("image_prompt"):
                if base:
                    try:
                        model = row.get("llm_model")
                        if not model:
                            model = DEFAULT_MODEL
                        kwargs = {}
                        for key in ("temperature", "max_tokens", "top_p"):
                            val = row.get(key)
                            if pd.notna(val) and val != "":
                                kwargs[key] = val
                        synopsis = build_image_prompt_context(row)
                        prompt = generate_prompt_for_row(
                            row,
                            synopsis,
                            model,
                            kwargs.get("temperature", DEFAULT_TEMPERATURE),
                            kwargs.get("max_tokens"),
                            kwargs.get("top_p"),
                            int(row.get("timeout", DEFAULT_TIMEOUT) or DEFAULT_TIMEOUT),
                        )
                        if prompt:
                            if nsfw:
                                prompt = f"{prompt.rstrip()} NSFW"
                            df.at[idx, "image_prompt"] = prompt
                            st.toast(
                                f"Prompt generated for row {row.get('id', idx)}"
                            )
                    except Exception as e:
                        st.error(
                            f"Prompt generation failed for row {row.get('id', idx)}: {e}"
                        )
            else:
                st.toast(
                    f"Row {row.get('id', idx)} already has an image_prompt",
                    icon="⚠️",
                )

        iterate_selected(df, process)
        st.session_state.image_df = df
        if st.session_state.autosave:
            save_data(df, CSV_FILE)
        rerun_with_message("Page reloaded after generating prompts")


    if gen_col.button("Generate images"):
        df = st.session_state.image_df

        def process(idx: int, row: pd.Series) -> None:
            prompt = row.get("image_prompt", "")
            if not prompt:
                st.warning(f"No image_prompt for row {row.get('id', idx)}")
                return
            neg_prompt = row.get("negative_prompt", "") or DEFAULT_NEGATIVE_PROMPT
            sfw_neg = row.get("sfw_negative_prompt", "")
            if row.get("nsfw"):
                final_neg = neg_prompt
            else:
                final_neg = ", ".join(filter(None, [neg_prompt, sfw_neg]))
            checkpoint = row.get("checkpoint") or ""
            if not checkpoint:
                if st.session_state.comfy_models:
                    checkpoint = st.session_state.comfy_models[0]
                else:
                    st.warning("No ComfyUI checkpoints available")
                    return
            vae = row.get("comfy_vae") or ""
            row_id = row.get("id", idx)
            seed = row.get("seed")
            seed = coerce_int(seed, DEFAULT_SEED, "seed", row_id, min_value=-1)
            if seed == -1:
                seed = random.randint(0, 2**32 - 1)
            steps_val = coerce_int(
                row.get("steps"), DEFAULT_STEPS, "steps", row_id, min_value=1
            )
            width_val = coerce_int(
                row.get("width"), DEFAULT_WIDTH, "width", row_id, min_value=1
            )
            height_val = coerce_int(
                row.get("height"), DEFAULT_HEIGHT, "height", row_id, min_value=1
            )
            cfg_val = coerce_float(
                row.get("cfg"), DEFAULT_CFG, "cfg", row_id, min_value=0
            )
            batch = coerce_int(
                row.get("batch_count"), DEFAULT_BATCH, "batch_count", row_id, min_value=1
            )
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cat_raw = str(row.get("category", "")).strip()
            tag_raw = str(row.get("tags", "")).replace(",", "_").strip()
            parts = []
            if cat_raw:
                parts.append(slugify(cat_raw))
            if tag_raw:
                parts.append(slugify(tag_raw))
            parts.extend([checkpoint, timestamp])
            folder = Path("items") / "_".join(parts)
            folder.mkdir(parents=True, exist_ok=True)

            for b in range(batch):
                try:
                    paths = generate_image(
                        prompt,
                        checkpoint=checkpoint,
                        vae=vae,
                        seed=seed + b,
                        negative_prompt=final_neg,
                        width=width_val,
                        height=height_val,
                        cfg=cfg_val,
                        steps=steps_val,
                        output_dir=folder,
                        prefix=f"batch{b}",
                        debug=DEBUG_MODE,
                    )
                    if paths:
                        for p in paths:
                            st.success(f"Saved image to {p}")
                    else:
                        st.error(
                            f"Failed to generate image for row {row.get('id', idx)}"
                        )
                except Exception as e:
                    st.error(
                        f"Image generation error for row {row.get('id', idx)}: {e}"
                    )

            df.at[idx, "image_path"] = str(folder.resolve())

        iterate_selected(df, process)
        st.session_state.image_df = df
        if st.session_state.autosave:
            save_data(df, CSV_FILE)
        rerun_with_message("Page reloaded after generating images")


    if post_col.button("Post"):
        df = st.session_state.image_df

        def process(idx: int, row: pd.Series) -> None:
            image_path = row.get("image_path", "")
            if not image_path or not os.path.exists(image_path):
                st.warning(f"No image file for row {row.get('id', idx)}")
                return
            try:
                result = post_to_wordpress(row)
                if result:
                    df.at[idx, "post_url"] = result["link"]
                    df.at[idx, "post_site"] = result["site"]
                    df.at[idx, "post_id"] = str(result["id"])
                    st.success(f"Posted: {result['link']}")
                    print(f"[INFO] Posted: {result['link']}")
            except Exception as e:
                message = f"Posting failed for row {row.get('id', idx)}: {e}"
                st.error(message)
                print(f"[ERROR] {message}")

        iterate_selected(df, process)
        st.session_state.image_df = df
        if st.session_state.autosave:
            save_data(df, CSV_FILE)
        rerun_with_message("Page reloaded after posting")


    if anal_col.button("Analysis"):
        df = st.session_state.image_df

        def process(idx: int, row: pd.Series) -> None:
            site = row.get("post_site", "")
            post_id = row.get("post_id", "")
            if not site or not post_id:
                st.warning(
                    f"No post_site/post_id for row {row.get('id', idx)}"
                )
                return
            try:
                counts = fetch_view_counts(site, post_id)
                for column, value in counts.items():
                    df.at[idx, column] = value
            except Exception as e:
                st.error(
                    f"Analysis failed for row {row.get('id', idx)}: {e}"
                )

        iterate_selected(df, process)
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
