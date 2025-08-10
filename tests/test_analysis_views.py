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
