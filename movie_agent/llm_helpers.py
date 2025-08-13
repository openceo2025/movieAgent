import pandas as pd
from typing import Optional

from .ollama import list_ollama_models, generate_story_prompt, DEBUG_MODE
from .lmstudio import list_lmstudio_models, generate_story_prompt_lmstudio


def select_llm_models(df: pd.DataFrame) -> list[str]:
    """Return available LLM models based on the ``llm_environment`` column."""
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


__all__ = ["select_llm_models", "generate_prompt_for_row"]
