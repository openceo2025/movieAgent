import subprocess
import requests

from movie_agent.ollama import list_ollama_models, generate_story_prompt


def test_list_ollama_models(monkeypatch):
    class Dummy:
        def __init__(self, stdout):
            self.stdout = stdout

    def fake_run(*args, **kwargs):
        return Dummy(
            (
                "NAME\tMODIFIED\tSIZE\n"
                "phi3:mini 2024-01-01 1GB\n"
                "llama2:7b 2024-01-02 2GB\n"
            )
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    models = list_ollama_models()
    assert models == ["phi3:mini", "llama2:7b"]


def test_generate_story_prompt(monkeypatch):
    captured = {}

    class FakeResponse:
        status_code = 200
        text = (
            '{"response": "<think>reasoning</think>Once upon a time", '
            '"analysis": "because of reasons"}'
        )

        def json(self):
            return {
                "response": "<think>reasoning</think>Once upon a time",
                "analysis": "because of reasons",
            }

        def raise_for_status(self):
            pass

    def fake_post(url, json=None, **kwargs):
        captured["url"] = url
        captured["json"] = json
        return FakeResponse()

    monkeypatch.setattr(requests, "post", fake_post)
    result = generate_story_prompt("A synopsis", "phi3:mini")
    assert result == "Once upon a time"
    assert captured["json"] == {
        "model": "phi3:mini",
        "prompt": "Generate a short story based on this context:\nA synopsis\n",
        "stream": False,
        "options": {"temperature": 0.8},
    }
