# Streamlit Video-Agent – Sheet-Driven Local AI Pipeline

## 1. 概要
本プロジェクトは **Streamlit** を UI とし、  
スプレッドシートで管理したプロンプト／メタデータから  
ローカルの **Ollama・ComfyUI・framepack** 等を逐次呼び出して  
動画生成〜各種 SNS への自動投稿までを一気通貫で実行する “エージェント” です。  
インターネット接続は投稿時のみ。生成処理はすべてローカルで完結します。

---

## 2. 主要コンポーネント

| レイヤ          | 役割 | 使用技術 |
|-----------------|------|----------|
| **UI**          | セル編集／進捗監視／承認フラグ | Streamlit + Data Editor |
| **データ管理**  | 1 行＝1 本の動画定義 | Google Sheets or local CSV (切替可) |
| **生成エンジン**| 画像・音声・動画 | Ollama (LLM), ComfyUI (画像), framepack (動画補間) |
| **パイプライン**| ステップ実行・依存管理 | Python asyncio + TaskQueue |
| **投稿モジュール**| SNS / YouTube など API 連携 | sns-poster ラッパー群 |

---

## 3. スプレッドシート構成

| 列名 | 説明 | 例 |
|------|------|----|
| `title` | 動画タイトル | 「10 分でわかる量子コンピュータ」 |
| `synopsis` | あらすじ／要約 | … |
| `story_prompt` | シナリオ生成プロンプト | … |
| `bgm_prompt` | BGM 生成 or 取得条件 | `lofi science` |
| `taste_prompt` | 絵柄・雰囲気 | `flat comic, high-contrast` |
| `character_voice` | キャラ音声設定 | `female, cheerful, JP` |
| `status` | 進捗ステータス | `sheet`, `panel`, `video`, `ready`, `posted` |
| `needs_approve` | 人的承認要否 (`Y/N`) | `Y` |

> **セル編集→保存** がトリガー。各行に対しステータスマシンが動き、完了・失敗を即時反映します。

---

## 4. 生成用フォルダ構造（row-id = 42 の例）
project_root/
└─ vids/
  └─ 0042_quantum_intro/
    ├─ metadata.yaml # シートの行を YAML 化
    ├─ panels/ # コマ割り PNG
    ├─ audio/
    │   ├─ bgm.wav
    │   └─ voice.wav
    ├─ video_raw.mp4 # framepack 出力
    ├─ video_final.mp4 # エンコード後
    └─ thumbs/
        └─ yt_cover.png 

*上記ファイルは自動生成。人間が直接差し替えても次ステップはそれを利用。*

---

## 5. ワークフロー

1. **コマ割り画像作成**  
   *Ollama→ComfyUI* で各シーンの静止画を生成。  
2. **補間・動画化**  
   *framepack* で 24 fps 動画へ。  
3. **編集作業**  
   クロスフェード・カットなど ffmpeg スクリプト。  
4. **音声合成**  
   キャラクター音声（Ollama TTS 等）と BGM をミックス。  
5. **テロップ挿入**  
   Whisper で自動字幕 → srt → burn-in。  
6. **BGM 調整**  
   ボリュームオートメーション & ループ。  
7. **投稿**  
   YouTube / TikTok / X (Twitter) など API で自動アップロード。  

各ステップ終了時、`status` 列が更新され、`needs_approve` が `Y` の場合は処理待ち。  
承認列を `OK` に書き換えると次ステップへ進みます。

---

## 6. ローカル実行方法

```bash
# 仮想環境
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

# 起動
streamlit run movie_agent/app.py
# -> ブラウザで http://localhost:8501/
```

## 7. 今後の拡張アイデア
Stable Diffusion XL → ComfyUI に差し替え時のワークフロー自動切替

Spreadsheet から Notion DB へ移行するコネクタ

投稿結果の KPI (views, likes) を行単位で自動取得 → A/B テスト




