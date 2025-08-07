import json

import pandas as pd
import requests
import streamlit as st

from movie_agent.image_ui import post_to_wordpress


def test_post_to_wordpress(monkeypatch, tmp_path):
    # create dummy images in temporary directory
    img_b = tmp_path / "b.png"
    img_b.write_bytes(b"second")
    img_a = tmp_path / "a.png"
    img_a.write_bytes(b"first")

    captured = {}

    class FakeResponse:
        def __init__(self, data):
            self.status_code = 200
            self.text = json.dumps(data)

        def raise_for_status(self):
            pass

        def json(self):
            return json.loads(self.text)

    def fake_post(url, *args, **kwargs):
        captured["payload"] = kwargs.get("json")
        return FakeResponse({"url": "https://example.com/post/1"})

    monkeypatch.setattr(requests, "post", fake_post)

    row = pd.Series({"category": "cats", "tags": "cute,funny", "image_path": str(tmp_path)})
    url = post_to_wordpress(row)
    row["post_url"] = url

    payload = captured["payload"]
    # Title should incorporate category
    assert payload["title"] == "毎日投稿AI生成画像 cats"
    # Tags should be joined as a comma-separated string
    assert payload["content"] == "cute, funny"
    # Media should list images in sorted order
    assert [m["filename"] for m in payload["media"]] == ["a.png", "b.png"]
    # Returned URL should be recorded to post_url
    assert row["post_url"] == "https://example.com/post/1"


def test_post_to_wordpress_http_error(monkeypatch, tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(b"first")

    class FakeResponse:
        status_code = 500
        text = "server error"

        def raise_for_status(self):
            raise requests.HTTPError("500 Server Error")

    def fake_post(url, *args, **kwargs):
        return FakeResponse()

    errors = []
    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(st, "error", lambda msg: errors.append(msg))

    row = pd.Series({"category": "cats", "tags": "cute", "image_path": str(tmp_path)})
    assert post_to_wordpress(row) is None
    assert errors and "WordPress投稿に失敗しました" in errors[0]


def test_post_to_wordpress_bad_status(monkeypatch, tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(b"first")

    class FakeResponse:
        status_code = 202
        text = "accepted"

        def raise_for_status(self):
            pass

        def json(self):
            return {}

    def fake_post(url, *args, **kwargs):
        return FakeResponse()

    errors = []
    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(st, "error", lambda msg: errors.append(msg))

    row = pd.Series({"category": "cats", "tags": "cute", "image_path": str(tmp_path)})
    assert post_to_wordpress(row) is None
    assert errors and "WordPress投稿に失敗しました" in errors[0]


def test_post_to_wordpress_no_url(monkeypatch, tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(b"first")

    class FakeResponse:
        status_code = 200
        text = json.dumps({})

        def raise_for_status(self):
            pass

        def json(self):
            return {}

    def fake_post(url, *args, **kwargs):
        return FakeResponse()

    warnings = []
    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(st, "warning", lambda msg: warnings.append(msg))

    row = pd.Series({"category": "cats", "tags": "cute", "image_path": str(tmp_path)})
    assert post_to_wordpress(row) is None
    assert warnings and "WordPressから投稿URLが返されませんでした" in warnings[0]
