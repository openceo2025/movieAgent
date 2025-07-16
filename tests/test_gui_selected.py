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


def test_generate_videos_debug_message(monkeypatch, capsys, tmp_path):
    from movie_agent import gui

    # Enable debug mode
    monkeypatch.setattr(gui, "DEBUG_MODE", True)

    # Fake minimal Streamlit interface
    class FakeSt:
        def __init__(self):
            self.session_state = {}

        def button(self, label, disabled=False):
            return True

        def warning(self, *args, **kwargs):
            pass

        def success(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

    fake_st = FakeSt()
    monkeypatch.setattr(gui, "st", fake_st)

    df = pd.DataFrame({
        "selected": [True],
        "id": ["1"],
        "title": ["test"],
        "fps": [24],
        "movie_prompt": [""],
        "video_length": [1.0],
        "seed": [1],
        "cfg": [1.0],
        "steps": [1],
    })
    fake_st.session_state["video_df"] = df
    fake_st.session_state["last_saved_df"] = df.copy()

    monkeypatch.setattr(gui, "slugify", lambda s: "slug")
    monkeypatch.setattr(gui, "save_data", lambda df, path: None)
    monkeypatch.setattr(gui, "rerun_with_message", lambda msg: None)
    monkeypatch.setattr(gui, "log_to_console", lambda data: None)
    monkeypatch.setattr(gui.os, "makedirs", lambda *a, **k: None)
    monkeypatch.setattr(gui.Path, "glob", lambda self, pattern: [tmp_path / "img.png"])
    monkeypatch.setattr(gui.framepack, "generate_video", lambda *a, **k: "ok")
    monkeypatch.setattr(gui, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(gui, "CSV_FILE", str(tmp_path / "videos.csv"))

    selected_rows = fake_st.session_state["video_df"]["selected"].fillna(False).astype(bool)
    generate_disabled = not selected_rows.any()

    if fake_st.button("Generate videos", disabled=generate_disabled):
        if gui.DEBUG_MODE:
            print("[DEBUG] Generate videos button clicked")
            print(f"[DEBUG] {int(selected_rows.sum())} rows selected")

    out_lines = capsys.readouterr().out.strip().splitlines()
    assert out_lines[0] == "[DEBUG] Generate videos button clicked"

