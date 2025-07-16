import os
from typing import Optional

from gradio_client import Client, handle_file

FRAMEPACK_HOST = os.getenv("FRAMEPACK_HOST", "127.0.0.1")
FRAMEPACK_PORT = os.getenv("FRAMEPACK_PORT", "8001")
FRAMEPACK_API_NAME = os.getenv("FRAMEPACK_API_NAME", "/validate_and_process")


def generate_video(
    image: str,
    prompt: str,
    seed: int,
    video_length: float,
    latent_window_size: int,
    steps: int,
    cfg: float,
    gs: float,
    rs: float,
    gpu_memory_preservation: float,
    use_teacache: bool,
    mp4_crf: int,
    debug: bool = False,
) -> Optional[str]:
    """Generate a video using the local framepack server.

    When ``debug`` is ``True`` the request URL and parameters along with the
    response are printed.
    """

    url = f"http://{FRAMEPACK_HOST}:{FRAMEPACK_PORT}/"
    client = Client(url)

    img_param = handle_file(image)

    if debug:
        print("[DEBUG] framepack url:", f"{url}{FRAMEPACK_API_NAME}")
        print(
            "[DEBUG] framepack params:",
            {
                "image": image,
                "prompt": prompt,
                "seed": seed,
                "video_length": video_length,
                "latent_window_size": latent_window_size,
                "steps": steps,
                "cfg": cfg,
                "gs": gs,
                "rs": rs,
                "gpu_memory_preservation": gpu_memory_preservation,
                "use_teacache": use_teacache,
                "mp4_crf": mp4_crf,
            },
        )

    try:
        result = client.predict(
            img_param,
            prompt,
            "",  # n_prompt
            seed,
            video_length,
            latent_window_size,
            steps,
            cfg,
            gs,
            rs,
            gpu_memory_preservation,
            use_teacache,
            mp4_crf,
            api_name=FRAMEPACK_API_NAME,
        )
        if debug:
            print("[DEBUG] framepack response:", result)
        return result
    except Exception as e:
        if debug:
            print("[DEBUG] framepack error:", e)
        return None


__all__ = [
    "generate_video",
    "FRAMEPACK_HOST",
    "FRAMEPACK_PORT",
    "FRAMEPACK_API_NAME",
]
