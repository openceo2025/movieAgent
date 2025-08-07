import argparse
import subprocess
import json
import re
import requests
import streamlit as st

# Parse optional debug flag when imported
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    "--debug",
    action="store_true",
    help="Enable debug logging",
)
args, _ = parser.parse_known_args()
DEBUG_MODE = args.debug
if DEBUG_MODE:
    print("[DEBUG] Debug mode enabled")


def list_ollama_models(debug: bool | None = None) -> list[str]:
    """Return available Ollama models."""
    if debug is None:
        debug = DEBUG_MODE
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
        if debug:
            print("[DEBUG] ollama list stdout:", result.stdout)
    except Exception as e:
        if debug:
            print("[DEBUG] ollama list error:", e)
        return ["gpt-oss:20b"]
    lines = result.stdout.strip().splitlines()
    models = []
    for line in lines[1:]:
        parts = line.split()
        if parts:
            models.append(parts[0])
    return models or ["gpt-oss:20b"]


def generate_story_prompt(
    synopsis: str,
    model: str,
    temperature: float = 0.8,
    max_tokens: int | None = None,
    top_p: float | None = None,
    debug: bool | None = None,
    timeout: int = 300,
) -> str | None:
    """Generate a story prompt using the local Ollama API."""
    if debug is None:
        debug = DEBUG_MODE
    prompt = f"Generate a short story based on this synopsis:\n{synopsis}\n"
    url = "http://localhost:11434/api/generate"
    payload: dict[str, object] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    options: dict[str, object] = {}
    if temperature is not None:
        options["temperature"] = temperature
    if max_tokens is not None:
        options["num_predict"] = max_tokens
    if top_p is not None:
        options["top_p"] = top_p
    if options:
        payload["options"] = options
    if debug:
        print("[DEBUG] Request payload:", payload)
    try:
        res = requests.post(url, json=payload, timeout=timeout, stream=False)
        if debug:
            print("[DEBUG] Response status:", res.status_code)
        res.raise_for_status()
        if debug:
            print("[DEBUG] Raw response:", res.text)
        data = res.json()
        if debug:
            print("[DEBUG] Parsed response:", json.dumps(data, indent=2))
        resp = data.get("response", "")
        reasoning_parts = []
        for key in ("thinking", "analysis"):
            if data.get(key):
                reasoning_parts.append(str(data[key]))
        think_matches = re.findall(r"<think>(.*?)</think>", resp, flags=re.DOTALL)
        if think_matches:
            reasoning_parts.extend([t.strip() for t in think_matches])
            resp = re.sub(r"<think>.*?</think>", "", resp, flags=re.DOTALL)
        resp = resp.strip()
        if debug and reasoning_parts:
            print("[Reasoning]", "\n".join(reasoning_parts).strip())
        return resp
    except requests.exceptions.RequestException as e:
        st.error(f"Ollama API error: {e}")
    except ValueError as e:
        st.error(f"Invalid response from Ollama: {e}")
    except Exception as e:
        st.error(f"Error: {e}")
    return None


__all__ = ["list_ollama_models", "generate_story_prompt", "DEBUG_MODE"]
