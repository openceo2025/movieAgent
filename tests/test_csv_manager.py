import os
import pandas as pd

from movie_agent.csv_manager import (
    load_data,
    load_image_data,
    save_data,
    unique_path,
    slugify,
    DEFAULT_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_CFG,
    DEFAULT_STEPS,
    DEFAULT_FPS,
    DEFAULT_VIDEO_LENGTH,
)


def test_load_data_missing(tmp_path):
    path = tmp_path / "missing.csv"
    df = load_data(path)
    assert df.empty
    assert "title" in df.columns
    # newly added columns should exist with defaults
    assert "movie_prompt" in df.columns
    assert "video_length" in df.columns
    assert "fps" in df.columns
    assert "batch_count" in df.columns
    assert "controlnet_image" in df.columns


def test_save_data(tmp_path):
    path = tmp_path / "out.csv"
    df = pd.DataFrame({"selected": [True], "id": ["1"], "title": ["test"]})
    save_data(df, path)
    loaded = pd.read_csv(path)
    assert "selected" not in loaded.columns
    assert str(loaded.loc[0, "id"]) == "1"


def test_unique_path(tmp_path):
    base = tmp_path / "file.txt"
    base.write_text("x")
    result1 = unique_path(str(base))
    assert result1.endswith("_1.txt")
    assert not os.path.exists(result1)
    # create the _1 file and ensure next call returns _2
    (tmp_path / "file_1.txt").write_text("y")
    result2 = unique_path(str(base))
    assert result2.endswith("_2.txt")
    assert not os.path.exists(result2)


def test_slugify_unicode():
    assert slugify("テスト 日本語") == "テスト_日本語"
    assert slugify("Hello 世界!!") == "hello_世界"


def test_load_data_defaults_existing_file(tmp_path):
    path = tmp_path / "data.csv"
    # CSV missing most columns, including temperature
    df = pd.DataFrame({"title": ["A"]})
    df.to_csv(path, index=False)
    loaded = load_data(path)
    assert loaded.loc[0, "title"] == "A"
    # temperature column should be filled with default
    assert loaded.loc[0, "temperature"] == DEFAULT_TEMPERATURE
    # other missing columns populated with defaults
    assert loaded.loc[0, "llm_model"] == DEFAULT_MODEL
    assert loaded.loc[0, "max_tokens"] == DEFAULT_MAX_TOKENS
    assert loaded.loc[0, "cfg"] == DEFAULT_CFG
    assert loaded.loc[0, "steps"] == DEFAULT_STEPS
    # selected column should default to False
    assert bool(loaded.loc[0, "selected"]) is False
    # new columns should have their defaults
    assert loaded.loc[0, "video_length"] == DEFAULT_VIDEO_LENGTH
    assert loaded.loc[0, "fps"] == DEFAULT_FPS
    assert loaded.loc[0, "movie_prompt"] == ""
    assert loaded.loc[0, "batch_count"] == 1
    assert loaded.loc[0, "controlnet_image"] == ""


def test_load_image_data_adds_post_columns(tmp_path):
    path = tmp_path / "img.csv"
    df = load_image_data(path)
    assert "post_site" in df.columns
    assert "post_id" in df.columns
    assert df["post_site"].eq("").all()
    assert df["post_id"].eq("").all()
    assert "wordpress_site" in df.columns
    assert df["wordpress_site"].eq("").all()
