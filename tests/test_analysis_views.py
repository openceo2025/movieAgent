import pandas as pd
import requests
import streamlit as st

from movie_agent.image_ui import AUTOPOSTER_API_URL


def test_analysis_updates_views(monkeypatch):
    df = pd.DataFrame(
        [
            {
                "selected": True,
                "post_site": "mysite",
                "post_id": 10,
                "views_yesterday": 0,
            }
        ]
    )
    st.session_state.image_df = df

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"views": [5]}

    def fake_get(url, params=None, timeout=10):
        return FakeResponse()

    monkeypatch.setattr(requests, "get", fake_get)

    selected = st.session_state.image_df[st.session_state.image_df["selected"]]
    for idx, row in selected.iterrows():
        site = row.get("post_site", "")
        post_id = row.get("post_id", "")
        res = requests.get(
            f"{AUTOPOSTER_API_URL}/wordpress/stats/views",
            params={"site": site, "post_id": post_id, "days": 1},
            timeout=10,
        )
        res.raise_for_status()
        data = res.json()
        st.session_state.image_df.at[idx, "views_yesterday"] = data.get("views", [0])[0]

    assert st.session_state.image_df.at[0, "views_yesterday"] == 5


def test_analysis_updates_week_and_month_views(monkeypatch):
    df = pd.DataFrame(
        [
            {
                "selected": True,
                "post_site": "mysite",
                "post_id": 10,
                "views_yesterday": 0,
                "views_week": 0,
                "views_month": 0,
            }
        ]
    )
    st.session_state.image_df = df

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

    selected = st.session_state.image_df[st.session_state.image_df["selected"]]
    for idx, row in selected.iterrows():
        site = row.get("post_site", "")
        post_id = row.get("post_id", "")
        for days, col in [
            (1, "views_yesterday"),
            (7, "views_week"),
            (30, "views_month"),
        ]:
            res = requests.get(
                f"{AUTOPOSTER_API_URL}/wordpress/stats/views",
                params={"site": site, "post_id": post_id, "days": days},
                timeout=10,
            )
            res.raise_for_status()
            data = res.json()
            views = data.get("views", [])
            st.session_state.image_df.at[idx, col] = views[0] if views else 0

    assert st.session_state.image_df.at[0, "views_week"] == 50
    assert st.session_state.image_df.at[0, "views_month"] == 300


def test_analysis_handles_empty_views(monkeypatch):
    df = pd.DataFrame(
        [
            {
                "selected": True,
                "post_site": "mysite",
                "post_id": 10,
                "views_yesterday": 1,
                "views_week": 1,
                "views_month": 1,
            }
        ]
    )
    st.session_state.image_df = df

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"views": []}

    def fake_get(url, params=None, timeout=10):
        return FakeResponse()

    monkeypatch.setattr(requests, "get", fake_get)

    selected = st.session_state.image_df[st.session_state.image_df["selected"]]
    for idx, row in selected.iterrows():
        site = row.get("post_site", "")
        post_id = row.get("post_id", "")
        for days, col in [
            (1, "views_yesterday"),
            (7, "views_week"),
            (30, "views_month"),
        ]:
            res = requests.get(
                f"{AUTOPOSTER_API_URL}/wordpress/stats/views",
                params={"site": site, "post_id": post_id, "days": days},
                timeout=10,
            )
            res.raise_for_status()
            data = res.json()
            views = data.get("views", [])
            st.session_state.image_df.at[idx, col] = views[0] if views else 0

    assert st.session_state.image_df.at[0, "views_yesterday"] == 0
    assert st.session_state.image_df.at[0, "views_week"] == 0
    assert st.session_state.image_df.at[0, "views_month"] == 0
