import pandas as pd
from movie_agent import gui


def test_select_llm_models_lmstudio(monkeypatch):
    df = pd.DataFrame({"llm_environment": ["LMStudio", "Ollama"]})
    monkeypatch.setattr(gui, "list_lmstudio_models", lambda: ["lm"])
    models = gui.select_llm_models(df)
    assert models == ["lm"]


def test_select_llm_models_ollama(monkeypatch):
    df = pd.DataFrame({"llm_environment": ["Ollama"]})
    monkeypatch.setattr(gui, "list_ollama_models", lambda: ["ol"])
    models = gui.select_llm_models(df)
    assert models == ["ol"]


def test_generate_prompt_for_row_lmstudio(monkeypatch):
    row = pd.Series({"llm_environment": "LMStudio"})
    monkeypatch.setattr(gui, "generate_story_prompt_lmstudio", lambda *args, **kwargs: "lm_resp")
    result = gui.generate_prompt_for_row(row, "ctx", "model", 0.5, 10, 0.9, 30)
    assert result == "lm_resp"


def test_generate_prompt_for_row_ollama(monkeypatch):
    row = pd.Series({"llm_environment": "Ollama"})
    monkeypatch.setattr(gui, "generate_story_prompt", lambda *args, **kwargs: "ol_resp")
    result = gui.generate_prompt_for_row(row, "ctx", "model", 0.5, 10, 0.9, 30)
    assert result == "ol_resp"
