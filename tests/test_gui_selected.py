import pandas as pd


def test_selected_nan_handled():
    df = pd.DataFrame({
        "selected": [True, float('nan'), False],
        "value": [1, 2, 3],
    })

    # Replicate GUI logic to sanitize the selected column
    df["selected"] = df["selected"].fillna(False).astype(bool)

    assert df["selected"].tolist() == [True, False, False]

    mask = df["selected"].fillna(False).astype(bool)
    filtered = df[mask]

    assert filtered["value"].tolist() == [1]

