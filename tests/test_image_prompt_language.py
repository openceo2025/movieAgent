import pandas as pd
from movie_agent.image_ui import build_image_prompt_context

def test_build_image_prompt_context_includes_language_hint_and_ethnic_instruction():
    row = pd.Series({
        "ja_prompt": "テストプロンプト",
        "nsfw": False,
        "id": "ja",
        "category": "風景",
        "tags": "自然",
    })
    result = build_image_prompt_context(row)
    assert "Language hint: ja" in result
    assert (
        "Ensure the prompt depicts people whose appearance reflects typical traits of regions where this language is primarily spoken."
        in result
    )
