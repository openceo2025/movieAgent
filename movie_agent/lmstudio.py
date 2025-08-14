"""Helpers for interacting with a local LM Studio server.

This module provides thin wrappers around LM Studio's OpenAI-compatible API.
The base URL is read from the ``LMSTUDIO_HOST`` environment variable and
defaults to ``http://localhost:1234``.
"""

from __future__ import annotations

import os
import requests
import streamlit as st

from movie_agent.logger import logger


def _base_url() -> str:
    """Return the base URL for LM Studio from the environment."""
    return os.environ.get("LMSTUDIO_HOST", "http://localhost:1234").rstrip("/")


def list_lmstudio_models(timeout: int | None = 30) -> list[str]:
    """Fetch available models from the LM Studio server."""
    url = f"{_base_url()}/v1/models"
    try:
        res = requests.get(url, timeout=timeout)
        res.raise_for_status()
        data = res.json()
        return [m.get("id", "") for m in data.get("data", []) if m.get("id")]
    except requests.exceptions.RequestException as e:
        st.error(f"LM Studio API error: {e}")
    except ValueError as e:
        st.error(f"Invalid response from LM Studio: {e}")
    return []


def generate_story_prompt_lmstudio(
    context: str,
    model: str,
    temperature: float = 0.8,
    max_tokens: int | None = None,
    top_p: float | None = None,
    timeout: int = 300,
) -> str | None:
    """Generate text using LM Studio's chat completions API."""

    url = f"{_base_url()}/v1/chat/completions"
    payload: dict[str, object] = {
        "model": model,
        "messages": [{"role": "user", "content": context}],
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if top_p is not None:
        payload["top_p"] = top_p

    try:
        res = requests.post(url, json=payload, timeout=timeout)
        res.raise_for_status()
        data = res.json()
        return data["choices"][0]["message"]["content"]
    except (requests.exceptions.RequestException, KeyError, IndexError, ValueError) as e:
        st.error(f"LM Studio API error: {e}")
        return None


def translate_with_lmstudio(
    text: str,
    lang: str,
    model: str = "",
    timeout: int = 300,
    log_prompt: bool = False,
) -> str | None:
    """Translate ``text`` into ``lang`` using LM Studio's chat API.

    If ``model`` is an empty string, the server's default model is used.
    Set ``log_prompt`` to ``True`` to log the generated prompt for debugging.
    """

    url = f"{_base_url()}/v1/chat/completions"
    prompt = (
        "翻訳ツールのように回答してくれ。余計な説明などはなく訳だけを答えてくれ。"
        f"この言葉「{text}」を言語「{lang}に記載されている文字列」に翻訳してくれ"
    )
    if log_prompt:
        logger.info(prompt)
    payload: dict[str, object] = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }
    if model:
        payload["model"] = model
    try:
        res = requests.post(url, json=payload, timeout=timeout)
        res.raise_for_status()
        data = res.json()
        return data["choices"][0]["message"]["content"]
    except (requests.exceptions.RequestException, KeyError, IndexError, ValueError) as e:
        st.error(f"LM Studio API error: {e}")
        return None


__all__ = [
    "list_lmstudio_models",
    "generate_story_prompt_lmstudio",
    "translate_with_lmstudio",
]

