import os
import json
import base64
import time
from pathlib import Path
from typing import List

import requests
import streamlit as st

# ComfyUI API host/port configuration
COMFYUI_HOST = os.getenv("COMFYUI_HOST", "127.0.0.1")
COMFYUI_PORT = os.getenv("COMFYUI_PORT", "8188")

# Default image generation settings
DEFAULT_CFG = 7
DEFAULT_STEPS = 28
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
DEFAULT_NEGATIVE_PROMPT = (
    "embedding:BadDream:0.6, embedding:BadHandsV2:0.4, "
    "blurry, watermark, lowres, jpeg artifacts"
)

# Utility coercion helpers


def _coerce_int(value, default, name, min_value=None):
    """Convert ``value`` to int or return ``default`` and toast on error."""
    if value in (None, ""):
        return default
    try:
        value = int(value)
    except (ValueError, TypeError):
        st.toast(f"Invalid {name}; using default {default}")
        return default
    if min_value is not None and value < min_value:
        st.toast(f"{name.capitalize()} must be >= {min_value}; using {default}")
        return default
    return value


def _coerce_float(value, default, name, min_value=None):
    """Convert ``value`` to float or return ``default`` and toast on error."""
    if value in (None, ""):
        return default
    try:
        value = float(value)
    except (ValueError, TypeError):
        st.toast(f"Invalid {name}; using default {default}")
        return default
    if min_value is not None and value < min_value:
        st.toast(f"{name.capitalize()} must be >= {min_value}; using {default}")
        return default
    return value

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
        "inputs": {
            "batch_size": 1,
            "height": DEFAULT_HEIGHT,
            "width": DEFAULT_WIDTH,
        },
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


def list_comfy_models() -> tuple[list[str], list[str], list[str]]:
    """Return available checkpoints, VAEs and LoRAs from ComfyUI."""
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
    cfg: float = DEFAULT_CFG,
    steps: int = DEFAULT_STEPS,
    control_image: str | None = None,
    output_dir: Path | str = Path("."),
    prefix: str = "image",
    debug: bool = False,
) -> List[Path]:
    """Generate image(s) via ComfyUI and save to ``output_dir``.

    Returns a list of file paths for the saved images. If generation fails,
    an empty list is returned.
    """
    seed = _coerce_int(seed, 0, "seed", min_value=0)
    width = _coerce_int(width, DEFAULT_WIDTH, "width", min_value=1)
    height = _coerce_int(height, DEFAULT_HEIGHT, "height", min_value=1)
    cfg = _coerce_float(cfg, DEFAULT_CFG, "cfg", min_value=0)
    steps = _coerce_int(steps, DEFAULT_STEPS, "steps", min_value=1)

    base = f"http://{COMFYUI_HOST}:{COMFYUI_PORT}"
    prompt_url = f"{base}/prompt"

    workflow = json.loads(json.dumps(BASE_WORKFLOW))
    workflow["6"]["inputs"]["text"] = prompt
    workflow["4"]["inputs"]["ckpt_name"] = checkpoint
    if vae:
        workflow["4"]["inputs"]["vae_name"] = vae
    workflow["3"]["inputs"]["seed"] = seed
    workflow["3"]["inputs"]["cfg"] = cfg
    workflow["3"]["inputs"]["steps"] = steps
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
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for _ in range(60):
        try:
            r = requests.get(history_url, timeout=10)
            r.raise_for_status()
            hist_resp = r.json()
            if debug:
                try:
                    preview = json.dumps(hist_resp)[:200]
                    print("[DEBUG] /history response:", preview)
                except Exception as e:
                    print("[DEBUG] error decoding history response:", e)
            hist = hist_resp.get(prompt_id, {})
            outputs = hist.get("outputs", {})
            for node_data in outputs.values():
                images = node_data.get("images")
                if images:
                    saved_paths: List[Path] = []
                    for n, img in enumerate(images):
                        params = {"filename": img.get("filename")}
                        if img.get("subfolder"):
                            params["subfolder"] = img.get("subfolder")
                        if img.get("type"):
                            params["type"] = img.get("type")
                        resp = requests.get(view_url, params=params, timeout=10)
                        resp.raise_for_status()
                        out_path = output_dir / f"{prefix}_{n}.png"
                        with open(out_path, "wb") as f:
                            f.write(resp.content)
                        saved_paths.append(out_path)
                    return saved_paths
        except requests.exceptions.RequestException as e:
            if debug:
                print("[DEBUG] polling error:", e)
        except Exception as e:
            if debug:
                print("[DEBUG] unexpected error:", e)
        time.sleep(5)

    st.error("Timed out waiting for ComfyUI output")
    return []


__all__ = [
    "list_comfy_models",
    "generate_image",
    "COMFYUI_HOST",
    "COMFYUI_PORT",
]
