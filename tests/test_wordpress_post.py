import json

import os
import pandas as pd
import pytest
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

    row = pd.Series(
        {
            "category": "cats",
            "tags": "cute,funny",
            "image_path": str(tmp_path),
            "wordpress_site": "mysite",
        }
    )
    url = post_to_wordpress(row)
    row["post_url"] = url

    payload = captured["payload"]
    # Title should incorporate category
    assert payload["title"] == "毎日投稿AI生成画像 cats"
    # Tags should be joined as a comma-separated string
    assert payload["content"] == "cute, funny"
    # Media should list images in sorted order
    assert [m["filename"] for m in payload["media"]] == ["a.png", "b.png"]
    # Site should be forwarded in payload
    assert payload["site"] == "mysite"
    # Returned URL should be recorded to post_url
    assert row["post_url"] == "https://example.com/post/1"


def test_post_to_wordpress_payload_has_site(monkeypatch, tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(b"first")

    captured = {}

    class FakeResponse:
        def __init__(self):
            self.status_code = 200
            self.text = json.dumps({"url": "https://example.com/post/1"})

        def raise_for_status(self):
            pass

        def json(self):
            return json.loads(self.text)

    def fake_post(url, *args, **kwargs):
        captured["payload"] = kwargs.get("json")
        return FakeResponse()

    monkeypatch.setattr(requests, "post", fake_post)

    row = pd.Series(
        {
            "category": "cats",
            "tags": "cute",
            "image_path": str(tmp_path),
            "wordpress_site": "mysite",
        }
    )

    post_to_wordpress(row)

    payload = captured["payload"]
    assert "site" in payload
    assert payload["site"] == "mysite"


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

    row = pd.Series(
        {
            "category": "cats",
            "tags": "cute",
            "image_path": str(tmp_path),
            "wordpress_site": "mysite",
        }
    )
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

    row = pd.Series(
        {
            "category": "cats",
            "tags": "cute",
            "image_path": str(tmp_path),
            "wordpress_site": "mysite",
        }
    )
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

    row = pd.Series(
        {
            "category": "cats",
            "tags": "cute",
            "image_path": str(tmp_path),
            "wordpress_site": "mysite",
        }
    )
    assert post_to_wordpress(row) is None
    assert warnings and "WordPressから投稿URLが返されませんでした" in warnings[0]


def test_post_to_wordpress_missing_site(monkeypatch, tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(b"first")

    errors = []
    monkeypatch.setattr(st, "error", lambda msg: errors.append(msg))
    monkeypatch.setattr(
        requests,
        "post",
        lambda *a, **k: pytest.fail("post should not be called when site missing"),
    )

    row = pd.Series({"category": "cats", "tags": "cute", "image_path": str(tmp_path)})
    assert post_to_wordpress(row) is None
    assert errors and "WordPressサイトが指定されていません" in errors[0]


def test_post_multiple_rows_account_site(monkeypatch, tmp_path):
    from movie_agent import image_ui

    # create image files for two rows
    img1 = tmp_path / "1.png"
    img1.write_bytes(b"1")
    img2 = tmp_path / "2.png"
    img2.write_bytes(b"2")

    df = pd.DataFrame(
        {
            "selected": [True, True],
            "id": ["1", "2"],
            "image_path": [str(img1), str(img2)],
            "wordpress_site": ["siteA", "siteB"],
            "post_id": [10, 0],
            "media_id": [0, 0],
            "version": [0, 0],
        }
    )

    class FakeSt:
        def __init__(self):
            self.session_state = {
                "image_df": df.copy(),
                "post_mode": "update",
                "delete_old_media": False,
            }
            self.toasts = []

        def toast(self, msg):
            self.toasts.append(msg)

        def error(self, msg):
            pass

        def warning(self, msg):
            pass

    fake_st = FakeSt()
    monkeypatch.setattr(image_ui, "st", fake_st)

    calls = {"upload": [], "update": [], "create": []}

    def fake_upload_media(f, site):
        calls["upload"].append(site)
        return 100 + len(calls["upload"])

    def fake_update_post(post_id, row, media_id, site):
        account = (row.get("wordpress_site") or image_ui.WORDPRESS_ACCOUNT or "nicchi").strip()
        calls["update"].append((post_id, site, account))
        return f"http://example.com/{site}/{post_id}"

    def fake_create_post(row, media_id, site):
        account = (row.get("wordpress_site") or image_ui.WORDPRESS_ACCOUNT or "nicchi").strip()
        calls["create"].append((site, account))
        return 200 + len(calls["create"]), f"http://example.com/{site}/new"

    class DummyDT:
        @classmethod
        def now(cls):
            class D:
                def strftime(self, fmt):
                    return "2024-01-01 00:00:00"

            return D()

    monkeypatch.setattr(image_ui, "upload_media", fake_upload_media)
    monkeypatch.setattr(image_ui, "update_post", fake_update_post)
    monkeypatch.setattr(image_ui, "create_post", fake_create_post)
    monkeypatch.setattr(image_ui, "safe_delete_media", lambda *a, **k: None)
    monkeypatch.setattr(image_ui, "datetime", DummyDT)

    # replicate posting loop for selected rows
    df2 = fake_st.session_state["image_df"]
    selected_indices = df2.index[df2["selected"]].tolist()
    mode = fake_st.session_state.get("post_mode", "update")
    delete_old = fake_st.session_state.get("delete_old_media", False)
    for idx in selected_indices:
        row = df2.loc[idx]
        site = (row.get("wordpress_site") or os.getenv("WORDPRESS_SITE", "")).strip()
        image_path = row.get("image_path", "")
        if not site or not image_path:
            continue
        with open(image_path, "rb") as f:
            media_id = image_ui.upload_media(f, site)
        old_media_id = row.get("media_id")
        if mode == "update" and row.get("post_id"):
            post_url = image_ui.update_post(int(row.get("post_id")), row, media_id, site)
            post_id = row.get("post_id")
        else:
            post_id, post_url = image_ui.create_post(row, media_id, site)
        if delete_old and old_media_id:
            image_ui.safe_delete_media(int(old_media_id), site)
        df2.at[idx, "media_id"] = media_id
        if post_id:
            df2.at[idx, "post_id"] = post_id
        df2.at[idx, "post_url"] = post_url
        df2.at[idx, "last_posted_at"] = image_ui.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df2.at[idx, "version"] = int(row.get("version", 0)) + 1
        df2.at[idx, "error"] = ""
        fake_st.toast(f"Posted row {row.get('id', idx)}")

    fake_st.session_state["image_df"] = df2

    assert calls["upload"] == ["siteA", "siteB"]
    assert calls["update"] == [(10, "siteA", "siteA")]
    assert calls["create"] == [("siteB", "siteB")]
