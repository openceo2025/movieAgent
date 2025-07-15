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

```bash
# 通常起動
streamlit run movie_agent/app.py
# デバッグ表示を有効化する場合
streamlit run movie_agent/app.py -- --debug
```

実行後、ブラウザで [http://localhost:8501](http://localhost:8501) を開くと UI が表示されます。


動画リストは **Streamlit Data Editor** を利用しており、`num_rows="dynamic"`
を指定することで、表から直接行を追加・削除できます。モデルの選択に加え、
`temperature`、`max_tokens`、`top_p`、`cfg`、`steps`、`seed`、`batch_count`、`width`、`height` といった生成パラメータも列として
編集可能です。
デフォルト値は `temperature=0.7`、`max_tokens=4096`、`top_p=0.95`、`cfg=7`、`steps=28`、`seed=1234`、`batch_count=1`、`width=1024`、`height=1024`
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

