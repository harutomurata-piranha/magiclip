# MagiClip 本番用イメージ（マネージドPaaS / Render・Railway・Fly.io 等で共通に使える）
FROM python:3.12-slim

# 動画処理に ffmpeg / ffprobe が必須。フォント(fonts/)とjanomeはリポジトリ/依存に同梱済み
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 依存を先に入れてビルドキャッシュを効かせる
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリ本体（fonts/ bgm/ テンプレ等を含む。.dockerignore で不要物は除外）
COPY . .

ENV APP_ENV=production \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# PaaS は $PORT を渡してくる（無ければ8080）。動画生成はバックグラウンドスレッドで走るので
# ワーカー1・スレッド複数・タイムアウト長めが安全。
EXPOSE 8080
CMD ["sh", "-c", "gunicorn server:app --bind 0.0.0.0:${PORT:-8080} --workers 1 --threads 4 --timeout 180"]
