# movieAgent

スプレッドシートで管理したプロンプトを基に、ローカル環境で動画を生成し、SNS へ自動投稿する Streamlit アプリです。

## 導入方法

Conda を利用して環境を構築します。`environment.yml` から環境を作成した後、
アクティブにしてください。

```bash
conda env create -f environment.yml
conda activate movieagent
```

## UI の起動方法

Windows 環境では付属の `start_movie_agent.bat` を実行してください。仮想環境
が存在しない場合は自動的に作成してから Streamlit を起動します。`movieagent`
conda 環境を利用している場合は、`start_movie_agent_conda.bat` を使用すると
`conda activate movieagent` を実行した後に同様の方法で UI を起動できます。画像
投稿用 UI を使う際は `start_image_ui.bat` または `start_image_ui_conda.bat` を
実行してください。

```bash
start_movie_agent.bat
REM または conda 環境用
start_movie_agent_conda.bat

start_image_ui.bat
REM 画像 UI を conda 環境で起動
start_image_ui_conda.bat
```

実行後、いずれもデフォルトのブラウザで [http://localhost:8501](http://localhost:8501)
を開くと UI が表示されます。


動画リストは **Streamlit Data Editor** を利用しており、`num_rows="dynamic"`
を指定することで、表から直接行を追加・削除できます。モデルの選択に加え、
`temperature`、`max_tokens`、`top_p`、`cfg`、`steps`、`seed`、`batch_count`、`width`、`height` といった生成パラメータや、動画用の `movie_prompt`、`video_length`、`fps` も列として
編集可能です。
デフォルト値は `temperature=0.7`、`max_tokens=4096`、`top_p=0.95`、`cfg=7`、`steps=28`、`seed=1234`、`batch_count=1`、`width=1024`、`height=1024`、`video_length=0`、`fps=24`
です。`seed` 欄を空欄にすると、画像生成時に毎回ランダムなシード値が送られ
ます。`-1` を入力した場合はその値をそのまま API へ渡し、ComfyUI 側でランダム
シードが採用されます。`batch_count` を空欄にすると 1 枚だけ生成します。

"Generate story prompts" ボタンを押すと、選択された行のプロンプト生成後に
CSV へ自動保存し、ページをリフレッシュして結果を即座に反映します。
アプリから `st.rerun()` が実行された際はリロード後にメッセージが表示され、
"Page reloaded after generating prompts" などの文言で通知されます。

## ComfyUI Path

ComfyUI は本リポジトリとは別ディレクトリにインストールしてください。
アプリケーションからは `COMFYUI_PATH` 環境変数で ComfyUI のルートパスを参照する想定です。

```bash
# 例
export COMFYUI_PATH=/path/to/ComfyUI
```

現時点ではまだ自動連携処理は実装されていませんが、今後この環境変数を利用して
ComfyUI を呼び出す予定です。

## FramePack サーバーの起動

FramePack は動画フレーム補間を行う Gradio サーバーです。ローカルで利用する場合は
以下のように公式リポジトリを取得して起動します。

```bash
# 例
git clone https://github.com/lllyasviel/FramePack.git
cd FramePack
pip install -r requirements.txt
python demo_gradio.py --port 8001
```

サーバーのホスト名とポートは `FRAMEPACK_HOST`、`FRAMEPACK_PORT` 環境変数で変更でき
ます。未設定時はそれぞれ `127.0.0.1` と `8001` が使われます。

フレームパックの API エンドポイントは `/validate_and_process` が標準です。
必要に応じて `FRAMEPACK_API_NAME` 環境変数で変更できます。旧バージョンでは
`/predict` や `/generate` が使われている場合があります。本アプリではまず
`/validate_and_process` を呼び出し、失敗した場合は自動的に `/predict`
へフォールバックし、それも失敗した場合は `fn_index` を指定して呼び出します。
利用したい `fn_index` は `FRAMEPACK_FN_INDEX` 環境変数で設定でき、
デフォルトは `1` です。

API が受け取る主な引数は次の 13 個です。

1. `image` – 開始フレーム画像
2. `prompt` – 生成プロンプト
3. `n_prompt` – ネガティブプロンプト
4. `seed`
5. `video_length` – 動画の秒数
6. `latent_window_size`
7. `steps`
8. `cfg`
9. `gs`
10. `rs`
11. `gpu_memory_preservation`
12. `use_teacache`
13. `mp4_crf`

デフォルトの `demo_gradio.py` はこれらすべての値が渡されることを前提としており、
フレーム画像のディレクトリだけを送る形では動作しません。

例として Python から直接呼び出す場合は次のようになります。

```python
from gradio_client import Client, handle_file

client = Client("http://127.0.0.1:8001/")
result = client.predict(
    handle_file("start.png"),
    "prompt text",
    "",  # n_prompt
    0,
    2.0,
    9,
    25,
    1.0,
    10.0,
    0.0,
    6.0,
    True,
    16,
    api_name="/validate_and_process",
)
```

## 動画生成の例

Streamlit UI でフレーム画像を用意した行を選択し、画面下部の **Generate videos** ボ
タンを押すと `vids/<id>_<slug>/video_raw.mp4` が生成されます。フレームパックサーバ
ーが起動している必要があります。
ここで `fps` 列の値がフレームパックへ渡され、指定されたフレームレートで動画が生成されます。`movie_prompt` と `video_length` は後段の編集工程で利用される予定です。

## Image Generation & Auto-Posting UI

This Streamlit UI manages image-centric posts. It lets you maintain category, tag, and NSFW prompts, translate them to English via **Ollama**, generate images with **ComfyUI**, automatically post them to a local **autoPoster** service, and fetch view statistics.

### Required columns
The sheet handled by this UI should include:

- `id`
- `category`
- `tags`
- `nsfw`
- `ja_prompt`
- `image_prompt`
- `image_path`
- `post_url`
- `views_yesterday`
- `views_week`
- `views_month`

Existing LLM and image parameter columns (e.g. model, temperature, steps, seed, width, height, etc.) can also be used.

### Button actions
- **Prompt** – use Ollama to convert `ja_prompt` to an English `image_prompt`.
- **Generate** – call ComfyUI to create an image at `image_path`.
- **Post** – upload the image via the local `autoPoster` API and store the `post_url`.
- **Analysis** – query the `autoPoster` API to fill `views_yesterday`, `views_week`, and `views_month`.

### Running
```bash
streamlit run movie_agent/image_ui.py
```

### Environment variables
Set environment variables so the UI can reach local services:

- `OLLAMA_HOST` – base URL of the Ollama API (default `http://localhost:11434`).
- `COMFYUI_API_URL` – endpoint for the ComfyUI REST API (default `http://127.0.0.1:8188`).
- `AUTOPOSTER_API_URL` – URL of the local autoPoster service for posting and analytics.

