# 🎬 MagiClip

**動画をアップロードするだけで、字幕付きのショート動画が完成する魔法のツール。**

長尺の動画から見どころを自動で抽出し、テンポよくカット・字幕付け・BGMミックスまで自動で行います。

---

## ✨ 主な機能

- 📤 動画をアップロードするだけで全自動編集（縦型ショート 1080×1920）
- 🤖 AIが見どころのシーンを自動選定（OpenAI Whisper + Claude）
- 📝 字幕を自動生成・校正して焼き込み（音声と同期）
- 🎞 シーン間のフェード、CC0 BGMの自動ミックス
- 🛠 （プロ向け / experimentブランチ）シーン選択・サムネ確認・字幕編集・テキストベース編集

## 🧩 技術構成

- バックエンド: Python / Flask
- 動画処理: ffmpeg
- 文字起こし: OpenAI Whisper (`whisper-1`)
- 構成・字幕校正: Anthropic Claude
- 字幕描画: Pillow（透過PNG）＋ ffmpeg overlay

---

## 🚀 ローカルでのセットアップ

### 1. 必要なもの
- Python 3.11 以上
- **ffmpeg**（必須）
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt install ffmpeg`

### 2. インストール
```bash
git clone <このリポジトリのURL>
cd video-ai
pip install -r requirements.txt
```

### 3. APIキーの設定
プロジェクト直下に `.env` を作成し、以下を記入します（`.env` はGitに含まれません）。
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

### 4. （任意）BGMの配置
`bgm/` フォルダにCC0（クレジット不要）のmp3を以下の名前で置くと自動でミックスされます。
`bright.mp3 / emotional.mp3 / tense.mp3 / stylish.mp3 / comical.mp3`（詳細は `bgm/README.md`）

### 5. 起動
```bash
python app.py
```
ブラウザで **http://127.0.0.1:5001** を開いて動画をアップロードしてください。

---

## ☁️ Render へのデプロイ

このリポジトリには `render.yaml` と `Dockerfile` が含まれています（ffmpegと日本語フォントを同梱するためDockerを使用）。

1. このリポジトリをGitHubにpush
2. [Render](https://render.com) で **New → Blueprint** を選び、リポジトリを指定（`render.yaml` を自動検出）
3. 環境変数 `OPENAI_API_KEY` と `ANTHROPIC_API_KEY` をダッシュボードで設定
4. デプロイ完了後、発行されたURLにアクセス

> メモ: 動画処理はメモリを使うため `plan: starter` 以上を推奨します。デプロイ環境のストレージは一時的（再起動で消える）なため、生成物の永続化が必要な場合はクラウドストレージ連携を追加してください。BGMファイルはGit管理外のため、デプロイ先で使うにはリポジトリに含めるか別途配置が必要です。

---

## 📄 ライセンス / クレジット
- BGMは各自がCC0等のクレジット不要音源を用意してください。
- 文字起こし・生成にOpenAI / Anthropic のAPIを使用します（利用料が発生します）。
