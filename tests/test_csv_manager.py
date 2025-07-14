from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from movie_agent.csv_manager import load_data, save_data


def test_load_data_missing(tmp_path):
    path = tmp_path / "missing.csv"
    df = load_data(path)
    assert df.empty
    assert "title" in df.columns


def test_save_data(tmp_path):
    path = tmp_path / "out.csv"
    df = pd.DataFrame({"selected": [True], "id": ["1"], "title": ["test"]})
    save_data(df, path)
    loaded = pd.read_csv(path)
    assert "selected" not in loaded.columns
    assert str(loaded.loc[0, "id"]) == "1"
