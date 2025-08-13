# TODO: Remove this test after refactoring is complete.
# 現在のカラム順を固定するための一時的なテストです。

from movie_agent import csv_manager


def test_image_csv_columns():
    df = csv_manager.load_image_data("images.csv")
    expected = [
        "selected",
        "id",
        "category",
        "tags",
        "nsfw",
        "ja_prompt",
        "llm_model",
        "llm_environment",
        "image_prompt",
        "negative_prompt",
        "sfw_negative_prompt",
        "image_path",
        "post_url",
        "post_site",
        "post_id",
        "wordpress_site",
        "wordpress_account",
        "views_yesterday",
        "views_week",
        "views_month",
        "checkpoint",
        "comfy_vae",
        "comfy_lora",
        "temperature",
        "max_tokens",
        "top_p",
        "cfg",
        "steps",
        "seed",
        "batch_count",
        "width",
        "height",
    ]
    assert list(df.columns) == expected
