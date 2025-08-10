import requests

from movie_agent.image_ui import fetch_view_counts


def test_fetch_view_counts(monkeypatch):
    class FakeResponse:
        def __init__(self, views):
            self._views = views

        def raise_for_status(self):
            pass

        def json(self):
            return {"views": self._views}

    def fake_get(url, params=None, timeout=10):
        days = params.get("days")
        if days == 1:
            return FakeResponse([5])
        if days == 7:
            return FakeResponse([50])
        if days == 30:
            return FakeResponse([300])
        return FakeResponse([])

    monkeypatch.setattr(requests, "get", fake_get)

    result = fetch_view_counts("mysite", 10)
    assert result["views_yesterday"] == 5
    assert result["views_week"] == 50
    assert result["views_month"] == 300


def test_fetch_view_counts_empty(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"views": []}

    def fake_get(url, params=None, timeout=10):
        return FakeResponse()

    monkeypatch.setattr(requests, "get", fake_get)

    result = fetch_view_counts("mysite", 10)
    assert result["views_yesterday"] == 0
    assert result["views_week"] == 0
    assert result["views_month"] == 0
