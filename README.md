# movieAgent

スプレッドシートで管理したプロンプトを基に、ローカル環境で動画を生成し、SNS へ自動投稿する Streamlit アプリです。

## 導入方法

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
```

## UI の起動方法

Windows 環境では付属の `start_movie_agent.bat` を実行してください。仮想環境の
作成と依存関係のインストールを行った後、デバッグモードで Streamlit を起動しま
す。

```bash
start_movie_agent.bat
```

実行後、ブラウザで [http://localhost:8501](http://localhost:8501) を開くと UI が表示されます。


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

## 動画生成の例

Streamlit UI でフレーム画像を用意した行を選択し、画面下部の **Generate videos** ボ
タンを押すと `vids/<id>_<slug>/video_raw.mp4` が生成されます。フレームパックサーバ
ーが起動している必要があります。
ここで `fps` 列の値がフレームパックへ渡され、指定されたフレームレートで動画が生成されます。`movie_prompt` と `video_length` は後段の編集工程で利用される予定です。
