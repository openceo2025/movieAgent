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
streamlit run app.py
```

実行後、ブラウザで [http://localhost:8501](http://localhost:8501) を開くと UI が表示されます。


動画リストは **Streamlit Data Editor** を利用しており、`num_rows="dynamic"`
を指定することで、表から直接行を追加・削除できます。モデルの選択に加え、
`temperature`、`max_tokens`、`top_p` といった生成パラメータも列として編集
可能です。
デフォルト値は `temperature=0.7`、`max_tokens=4096`、`top_p=0.95` です。

"Generate story prompts" ボタンを押すと、選択された行のプロンプト生成後に
CSV へ自動保存し、ページをリフレッシュして結果を即座に反映します。
アプリから `st.rerun()` が実行された際はリロード後にメッセージが表示され、
"Page reloaded after generating prompts" などの文言で通知されます。
