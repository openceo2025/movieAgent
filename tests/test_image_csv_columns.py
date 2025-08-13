# TODO: Remove this test after refactoring is complete.
# 現在のカラム順を固定するための一時的なテストです。

from movie_agent import csv_manager
from movie_agent.csv_schema import IMAGE_COLUMNS


def test_image_csv_columns():
    df = csv_manager.load_image_data("images.csv")
    assert list(df.columns) == IMAGE_COLUMNS
