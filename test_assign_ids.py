import pandas as pd
from app import assign_ids


def test_assign_ids_fill_gap():
    df = pd.DataFrame({"id": ["0001", "0003"], "title": ["A", "C"]})
    new_row = pd.DataFrame([{"id": "", "title": "B"}])
    df = pd.concat([df.iloc[:1], new_row, df.iloc[1:]], ignore_index=True)
    result = assign_ids(df)
    assert result.loc[1, "id"] == "0002"


def test_assign_ids_append():
    df = pd.DataFrame({"id": ["0001", "0002"], "title": ["A", "B"]})
    df = pd.concat([df, pd.DataFrame([{"id": "", "title": "C"}])], ignore_index=True)
    result = assign_ids(df)
    assert result.loc[2, "id"] == "0003"

