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

### トラブルシューティング

`ModuleNotFoundError: No module named 'st_aggrid'` が表示された場合は、依存パッケージが不足しています。以下のコマンドで追加インストールしてから再度起動してください。

```bash
pip install streamlit-aggrid
```
