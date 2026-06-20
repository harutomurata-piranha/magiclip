# MagiClip - 動画AI編集ツール
FROM python:3.11-slim

# ffmpeg（動画編集）と 日本語フォント（字幕用 Noto CJK）をインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 字幕に使う日本語フォント（Linux: Noto CJK）
ENV SUBTITLE_FONT=/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc
# Renderが渡す $PORT で待ち受ける。ジョブはメモリ共有が必要なため worker は1つ、同時アクセスは threads で捌く
CMD gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 600
