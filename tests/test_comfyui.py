import json
import time
import requests

import movie_agent.comfyui as comfyui


def test_list_comfy_models(monkeypatch):
    class FakeResponse:
        def __init__(self, data):
            self.status_code = 200
            self._data = data
            self.text = json.dumps(data)

        def json(self):
            return self._data

        def raise_for_status(self):
            pass

    def fake_get(url, *args, **kwargs):
        if url.endswith("/models"):
            return FakeResponse(["checkpoints", "vae", "loras"])
        if url.endswith("/models/checkpoints"):
            return FakeResponse(["sd1"])
        if url.endswith("/models/vae"):
            return FakeResponse(["vae1"])
        if url.endswith("/models/loras"):
            return FakeResponse(["lora1"])
        raise AssertionError(url)

    monkeypatch.setattr(requests, "get", fake_get)
    ckpts, vaes, loras = comfyui.list_comfy_models()
    assert ckpts == ["sd1"]
    assert vaes == ["", "vae1"]
    assert loras == ["", "lora1"]


def test_generate_image(monkeypatch, tmp_path):
    class FakeResponse:
        def __init__(self, data=None, content=b""):
            self.status_code = 200
            self._data = data
            self.text = json.dumps(data) if data is not None else ""
            self.content = content

        def json(self):
            return self._data

        def raise_for_status(self):
            pass

    def fake_post(url, *args, **kwargs):
        assert url.endswith("/prompt")
        return FakeResponse({"prompt_id": "abc"})

    def fake_get(url, *args, **kwargs):
        if url.endswith("/history/abc"):
            return FakeResponse(
                {
                    "abc": {
                        "outputs": {"9": {"images": [{"filename": "img.png"}]}}
                    }
                }
            )
        if url.endswith("/view"):
            return FakeResponse(content=b"image-bytes")
        raise AssertionError(url)

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(time, "sleep", lambda x: None)

    paths = comfyui.generate_image(
        "p",
        "ckpt",
        "",
        1,
        cfg=5,
        steps=20,
        output_dir=tmp_path,
        prefix="img",
    )
    assert paths == [tmp_path / "img_0.png"]
    assert (tmp_path / "img_0.png").read_bytes() == b"image-bytes"


def test_generate_image_invalid_numbers(monkeypatch, tmp_path):
    class FakeResponse:
        def __init__(self, data=None, content=b""):
            self.status_code = 200
            self._data = data
            self.text = json.dumps(data) if data is not None else ""
            self.content = content

        def json(self):
            return self._data

        def raise_for_status(self):
            pass

    captured = {}

    def fake_post(url, *args, **kwargs):
        captured["json"] = kwargs.get("json")
        return FakeResponse({"prompt_id": "abc"})

    def fake_get(url, *args, **kwargs):
        if url.endswith("/history/abc"):
            return FakeResponse(
                {
                    "abc": {
                        "outputs": {"9": {"images": [{"filename": "img.png"}]}}
                    }
                }
            )
        if url.endswith("/view"):
            return FakeResponse(content=b"image-bytes")
        raise AssertionError(url)

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(time, "sleep", lambda x: None)
    monkeypatch.setattr(comfyui.st, "toast", lambda msg: None)

    comfyui.generate_image(
        "p",
        "ckpt",
        "",
        "badseed",
        width="w",
        height=None,
        cfg="oops",
        steps="n/a",
        output_dir=tmp_path,
        prefix="img",
    )

    payload = captured["json"]["prompt"]
    assert payload["3"]["inputs"]["cfg"] == comfyui.DEFAULT_CFG
    assert payload["3"]["inputs"]["steps"] == comfyui.DEFAULT_STEPS
    assert payload["3"]["inputs"]["seed"] == 0
    assert payload["5"]["inputs"]["width"] == comfyui.DEFAULT_WIDTH
    assert payload["5"]["inputs"]["height"] == comfyui.DEFAULT_HEIGHT
