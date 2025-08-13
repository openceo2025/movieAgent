from __future__ import annotations

from typing import Callable

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
