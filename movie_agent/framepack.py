import os
from typing import Optional

from gradio_client import Client

FRAMEPACK_HOST = os.getenv("FRAMEPACK_HOST", "127.0.0.1")
FRAMEPACK_PORT = os.getenv("FRAMEPACK_PORT", "8001")


def generate_video(
    frames_dir: str,
    fps: int = 24,
    output: str = "video.mp4",
    debug: bool = False,
) -> Optional[str]:
    """Generate a video from ``frames_dir`` using the local framepack server."""
    url = f"http://{FRAMEPACK_HOST}:{FRAMEPACK_PORT}/"
    client = Client(url)
    try:
        result = client.predict(frames_dir, fps, output, api_name="/predict")
        if debug:
            print("[DEBUG] framepack response:", result)
        return result
    except Exception as e:
        if debug:
            print("[DEBUG] framepack error:", e)
        return None


__all__ = ["generate_video", "FRAMEPACK_HOST", "FRAMEPACK_PORT"]
