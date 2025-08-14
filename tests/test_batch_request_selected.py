import pandas as pd
from movie_agent.row_utils import batch_request_selected

def test_batch_request_selected(monkeypatch):
    df = pd.DataFrame([
        {"selected": True, "slug": "a"},
        {"selected": True, "slug": "b"},
    ])

    def prompt_builder(row: pd.Series) -> str:  # type: ignore[name-defined]
        return f"ctx-{row['slug']}"

    call_count = {"count": 0}

    def request_fn(prompts):
        call_count["count"] += 1
        assert prompts == ["ctx-a", "ctx-b"]
        return ["r1", "r2"]

    cache: dict[str, str] = {}
    batch_request_selected(df, prompt_builder, request_fn, cache)
    assert call_count["count"] == 1
    assert cache["ctx-a"] == "r1"
    assert cache["ctx-b"] == "r2"

    batch_request_selected(df, prompt_builder, request_fn, cache)
    assert call_count["count"] == 1
