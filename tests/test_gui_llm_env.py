import pandas as pd
from movie_agent import llm_helpers


def test_select_llm_models_lmstudio(monkeypatch):
    df = pd.DataFrame({"llm_environment": ["LMStudio", "Ollama"]})
    monkeypatch.setattr(llm_helpers, "list_lmstudio_models", lambda: ["lm"])
    models = llm_helpers.select_llm_models(df)
    assert models == ["lm"]


def test_select_llm_models_ollama(monkeypatch):
    df = pd.DataFrame({"llm_environment": ["Ollama"]})
    monkeypatch.setattr(llm_helpers, "list_ollama_models", lambda: ["ol"])
    models = llm_helpers.select_llm_models(df)
    assert models == ["ol"]


def test_generate_prompt_for_row_lmstudio(monkeypatch):
    row = pd.Series({"llm_environment": "LMStudio"})
    monkeypatch.setattr(
        llm_helpers, "generate_story_prompt_lmstudio", lambda *args, **kwargs: "lm_resp"
    )
    result = llm_helpers.generate_prompt_for_row(row, "ctx", "model", 0.5, 10, 0.9, 30)
    assert result == "lm_resp"


def test_generate_prompt_for_row_ollama(monkeypatch):
    row = pd.Series({"llm_environment": "Ollama"})
    monkeypatch.setattr(
        llm_helpers, "generate_story_prompt", lambda *args, **kwargs: "ol_resp"
    )
    result = llm_helpers.generate_prompt_for_row(row, "ctx", "model", 0.5, 10, 0.9, 30)
    assert result == "ol_resp"
