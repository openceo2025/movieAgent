from pathlib import Path
import sys
import subprocess
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from movie_agent.ollama import list_ollama_models, generate_story_prompt


def test_list_ollama_models(monkeypatch):
    class Dummy:
        def __init__(self, stdout):
            self.stdout = stdout

    def fake_run(*args, **kwargs):
        return Dummy("NAME\tMODIFIED\tSIZE\nphi3:mini 2024-01-01 1GB\nllama2:7b 2024-01-02 2GB\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    models = list_ollama_models()
    assert models == ["phi3:mini", "llama2:7b"]


def test_generate_story_prompt(monkeypatch):
    class FakeResponse:
        status_code = 200
        text = '{"response": "Once upon a time"}'

        def json(self):
            return {"response": "Once upon a time"}

        def raise_for_status(self):
            pass

    def fake_post(*args, **kwargs):
        return FakeResponse()

    monkeypatch.setattr(requests, "post", fake_post)
    result = generate_story_prompt("A synopsis", "phi3:mini", 0.7, 10, 0.9)
    assert result == "Once upon a time"

