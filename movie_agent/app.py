import streamlit as st
import pandas as pd
import subprocess
import requests
import argparse
import sys
import os
import json
import base64
import time
import random
from pathlib import Path
from typing import Optional

# Parse CLI arguments passed after `--` when running via Streamlit
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--debug", action="store_true", help="Enable debug logging")
args, _ = parser.parse_known_args()
DEBUG_MODE = args.debug
if DEBUG_MODE:
    print("[DEBUG] Debug mode enabled")

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_FILE = str(BASE_DIR / "videos.csv")

# Default generation parameters
DEFAULT_MODEL = "phi3:mini"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TOP_P = 0.95
DEFAULT_SEED = 1234
DEFAULT_NEGATIVE_PROMPT = (
    "embedding:BadDream:0.6, embedding:BadHandsV2:0.4, "
    "blurry, watermark, lowres, jpeg artifacts"
)

# Default image settings for ComfyUI
DEFAULT_CFG = 7
DEFAULT_STEPS = 28
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024

# ComfyUI API host/port
COMFYUI_HOST = os.getenv("COMFYUI_HOST", "127.0.0.1")
COMFYUI_PORT = os.getenv("COMFYUI_PORT", "8188")

# Base workflow used for the ComfyUI /prompt API
BASE_WORKFLOW = {
    "3": {
        "class_type": "KSampler",
        "inputs": {
            "cfg": DEFAULT_CFG,
            "denoise": 1,
            "latent_image": ["5", 0],
            "model": ["4", 0],
            "negative": ["7", 0],
            "positive": ["6", 0],
            "sampler_name": "euler",
            "scheduler": "karras",
            "seed": 8566257,
            "steps": DEFAULT_STEPS,
        },
    },
    "4": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": "sdXL_v10VAEFix.safetensors"},
    },
    "5": {
        "class_type": "EmptyLatentImage",
        "inputs": {"batch_size": 1, "height": DEFAULT_HEIGHT, "width": DEFAULT_WIDTH},
    },
    "6": {
        "class_type": "CLIPTextEncode",
        "inputs": {"clip": ["4", 1], "text": ""},
    },
    "7": {
        "class_type": "CLIPTextEncode",
        "inputs": {"clip": ["4", 1], "text": DEFAULT_NEGATIVE_PROMPT},
    },
    "8": {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
    },
    "9": {
        "class_type": "SaveImage",
        "inputs": {"filename_prefix": "ComfyUI", "images": ["8", 0]},
    },
}


from .csv_manager import (
    load_data,
    save_data,
    assign_ids,
    slugify,
    unique_path,
)


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
    """Return (checkpoints, vae, loras) from ComfyUI.

    A blank string is prepended to the VAE and LoRA lists so users can keep
    these fields empty if desired.
    """
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
    vaes = [""] + fetch("vae")
    loras = [""] + fetch("loras")
    return checkpoints, vaes, loras


def generate_image(
    prompt: str,
    checkpoint: str,
    vae: str,
    seed: int,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    control_image: str | None = None,
    debug: bool = False,
) -> Optional[bytes]:
    """Generate image via ComfyUI using polling."""
    base = f"http://{COMFYUI_HOST}:{COMFYUI_PORT}"
    prompt_url = f"{base}/prompt"

    workflow = json.loads(json.dumps(BASE_WORKFLOW))
    workflow["6"]["inputs"]["text"] = prompt
    workflow["4"]["inputs"]["ckpt_name"] = checkpoint
    if vae:
        workflow["4"]["inputs"]["vae_name"] = vae
    workflow["3"]["inputs"]["seed"] = seed
    workflow["5"]["inputs"]["width"] = width
    workflow["5"]["inputs"]["height"] = height

    if control_image and isinstance(control_image, str):
        encoded = base64.b64encode(open(control_image, "rb").read()).decode()
        workflow["10"] = {
            "class_type": "LoadImage",
            "inputs": {"image": encoded},
        }
        workflow["11"] = {
            "class_type": "ControlNetLoader",
            "inputs": {"image": ["10", 0]},
        }
        workflow["3"]["inputs"]["control_net"] = ["11", 0]

    payload = {"prompt": workflow}

    if debug:
        print("[DEBUG] ComfyUI request payload:", payload)

    try:
        res = requests.post(prompt_url, json=payload, timeout=30)
        if debug:
            print("[DEBUG] /prompt status:", res.status_code)
            print("[DEBUG] /prompt raw response:", res.text[:200])
        res.raise_for_status()
        data = res.json()
        prompt_id = data.get("prompt_id")
        if not prompt_id:
            st.error("prompt_id not returned from ComfyUI")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"ComfyUI API error: {e}")
        return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None

    history_url = f"{base}/history/{prompt_id}"
    view_url = f"{base}/view"
    for _ in range(60):
        try:
            r = requests.get(history_url, timeout=10)
            r.raise_for_status()
            hist_resp = r.json()
            if debug:
                try:
                    print(
                        "[DEBUG] /history response:",
                        json.dumps(hist_resp)[:200],
                    )
                except Exception as e:
                    print("[DEBUG] error decoding history response:", e)
            hist = hist_resp.get(prompt_id, {})
            outputs = hist.get("outputs", {})
            for node_data in outputs.values():
                images = node_data.get("images")
                if images:
                    img = images[0]
                    params = {"filename": img.get("filename")}
                    if img.get("subfolder"):
                        params["subfolder"] = img.get("subfolder")
                    if img.get("type"):
                        params["type"] = img.get("type")
                    resp = requests.get(view_url, params=params, timeout=10)
                    resp.raise_for_status()
                    return resp.content
        except requests.exceptions.RequestException as e:
            if debug:
                print("[DEBUG] polling error:", e)
        except Exception as e:
            if debug:
                print("[DEBUG] unexpected error:", e)
        time.sleep(5)

    st.error("Timed out waiting for ComfyUI output")
    return None


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
df_display = st.session_state.video_df.drop(columns=["controlnet_image"], errors="ignore")

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
            default=st.session_state.comfy_models[0] if st.session_state.comfy_models else "",
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
    },
    num_rows="dynamic",
    hide_index=True,
    use_container_width=True,
    key="video_editor",
)
new_df = edited_df.copy()
for col in df_display.columns:
    st.session_state.video_df[col] = new_df[col]
if st.session_state.autosave and not st.session_state.video_df.equals(
    st.session_state.last_saved_df
):
    save_data(st.session_state.video_df, CSV_FILE)
    st.session_state.last_saved_df = st.session_state.video_df.copy()
    st.info("Auto-saved to CSV")

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

        for b in range(batch_count):
            if seed_val == -1:
                # -1 is passed through so ComfyUI handles randomization
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
                st.error(
                    f"Failed to generate image for row {row.get('id', idx)} (batch {b+1})"
                )

    st.session_state.video_df = df
    save_data(df, CSV_FILE)
    st.session_state.last_saved_df = df.copy()
    rerun_with_message("Page reloaded after generating images")

save_col, auto_col = st.columns([1, 1])
auto_col.checkbox("Auto-save", value=st.session_state.autosave, key="autosave")

if save_col.button("Save CSV"):
    df = st.session_state.video_df.copy()
    st.session_state.video_df = df
    save_data(df, CSV_FILE)
    st.session_state.last_saved_df = df.copy()
    st.success("Saved to CSV")
