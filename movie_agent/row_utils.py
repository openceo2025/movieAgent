from __future__ import annotations

from typing import Callable, Any, List

import pandas as pd


def iterate_selected(df: pd.DataFrame, callback: Callable[[int, pd.Series], None]) -> None:
    """Iterate over rows where the ``selected`` column is truthy.

    The ``selected`` column may contain missing values.  This helper
    normalizes the column by treating NaN as ``False`` and casting to
    ``bool`` before iterating.  For each selected row, ``callback`` is
    invoked with ``(index, row)`` similar to ``DataFrame.iterrows``.
    """
    if "selected" not in df.columns:
        return

    mask = df["selected"].fillna(False).astype(bool)
    for idx, row in df[mask].iterrows():
        callback(idx, row)


def batch_request_selected(
    df: pd.DataFrame,
    prompt_builder: Callable[[pd.Series], str],
    request_fn: Callable[[List[str]], List[Any]],
    cache: dict[str, Any],
) -> None:
    """Fetch LLM responses for selected rows in a single batch.

    ``prompt_builder`` constructs the prompt string for each row.  ``request_fn``
    receives a list of prompts and must return a list of results in the same
    order.  Any prompts already present in ``cache`` are skipped.  Newly fetched
    results are stored in ``cache`` keyed by the prompt string.
    """
    if "selected" not in df.columns:
        return

    mask = df["selected"].fillna(False).astype(bool)
    prompts: List[str] = []
    for _, row in df[mask].iterrows():
        prompt = prompt_builder(row)
        if prompt not in cache:
            prompts.append(prompt)

    if not prompts:
        return

    results = request_fn(prompts)
    for prompt, result in zip(prompts, results):
        cache[prompt] = result


__all__ = ["iterate_selected", "batch_request_selected"]
