import pandas as pd

from movie_agent.csv_manager import load_image_data
from movie_agent import llm_helpers


def test_load_image_data_sets_llm_environment(tmp_path):
    path = tmp_path / "img.csv"
    df = load_image_data(path)
    assert "llm_environment" in df.columns
    assert df["llm_environment"].eq("Ollama").all()


def test_generate_prompt_for_row_lmstudio(monkeypatch):
    row = pd.Series({"llm_environment": "LMStudio"})
    calls = {}

    def fake_lmstudio(*args, **kwargs):
        calls["lmstudio"] = True
        return "lm"

    def fake_ollama(*args, **kwargs):
        calls["ollama"] = True
        return "ol"

    monkeypatch.setattr(llm_helpers, "generate_story_prompt_lmstudio", fake_lmstudio)
    monkeypatch.setattr(llm_helpers, "generate_story_prompt", fake_ollama)

    result = llm_helpers.generate_prompt_for_row(row, "ctx", "model", 0.5, 10, 0.9, 30)
    assert result == "lm"
    assert calls.get("lmstudio")
    assert "ollama" not in calls


def test_generate_prompt_for_row_ollama(monkeypatch):
    row = pd.Series({"llm_environment": "Ollama"})
    calls = {}

    def fake_lmstudio(*args, **kwargs):
        calls["lmstudio"] = True
        return "lm"

    def fake_ollama(*args, **kwargs):
        calls["ollama"] = True
        return "ol"

    monkeypatch.setattr(llm_helpers, "generate_story_prompt_lmstudio", fake_lmstudio)
    monkeypatch.setattr(llm_helpers, "generate_story_prompt", fake_ollama)

    result = llm_helpers.generate_prompt_for_row(row, "ctx", "model", 0.5, 10, 0.9, 30)
    assert result == "ol"
    assert calls.get("ollama")
    assert "lmstudio" not in calls
