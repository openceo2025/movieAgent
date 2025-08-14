import requests

from movie_agent.lmstudio import (
    list_lmstudio_models,
    generate_story_prompt_lmstudio,
    translate_with_lmstudio,
)


def test_list_lmstudio_models(monkeypatch):
    captured = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"data": [{"id": "phi"}, {"id": "llama"}]}

        def raise_for_status(self):
            pass

    def fake_get(url, timeout=None):
        captured["url"] = url
        return FakeResponse()

    monkeypatch.setenv("LMSTUDIO_HOST", "http://example.com")
    monkeypatch.setattr(requests, "get", fake_get)

    models = list_lmstudio_models()
    assert captured["url"] == "http://example.com/v1/models"
    assert models == ["phi", "llama"]


def test_generate_story_prompt_lmstudio(monkeypatch):
    captured = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": "Once upon a time"}}]}

        def raise_for_status(self):
            pass

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return FakeResponse()

    monkeypatch.setenv("LMSTUDIO_HOST", "http://example.com")
    monkeypatch.setattr(requests, "post", fake_post)

    result = generate_story_prompt_lmstudio(
        "context",
        "modelA",
        temperature=0.5,
        max_tokens=10,
        top_p=0.9,
    )
    assert captured["url"] == "http://example.com/v1/chat/completions"
    assert captured["json"] == {
        "model": "modelA",
        "messages": [{"role": "user", "content": "context"}],
        "temperature": 0.5,
        "max_tokens": 10,
        "top_p": 0.9,
    }
    assert result == "Once upon a time"


def test_translate_with_lmstudio(monkeypatch):
    captured = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {
                "choices": [{"message": {"content": "bonjour"}}]
            }

        def raise_for_status(self):
            pass

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return FakeResponse()

    monkeypatch.setenv("LMSTUDIO_HOST", "http://example.com")
    monkeypatch.setattr(requests, "post", fake_post)

    result = translate_with_lmstudio("hello", "fr")
    assert captured["url"] == "http://example.com/v1/chat/completions"
    assert "hello" in captured["json"]["messages"][0]["content"]
    assert "fr" in captured["json"]["messages"][0]["content"]
    assert "model" not in captured["json"]
    assert result == "bonjour"

