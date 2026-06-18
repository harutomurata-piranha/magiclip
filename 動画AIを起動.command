#!/bin/bash
# ダブルクリックで動画AIツールを起動し、ブラウザを開きます。
cd "$(dirname "$0")"

URL="http://127.0.0.1:5000"

# すでに起動しているか確認
if curl -s -o /dev/null "$URL" 2>/dev/null; then
  echo "✅ すでに起動しています。ブラウザを開きます…"
else
  echo "🚀 動画AIツールを起動しています…"
  nohup python3 app.py > server.log 2>&1 &
  # 起動するまで最大25秒待つ
  for i in $(seq 1 25); do
    if curl -s -o /dev/null "$URL" 2>/dev/null; then break; fi
    sleep 1
  done
fi

open "$URL"
echo "🎬 ブラウザで $URL を開きました。動画をアップロードしてください。"
echo "（このウィンドウは閉じてOKです。ツールはバックグラウンドで動き続けます）"
