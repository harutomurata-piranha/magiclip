#!/bin/bash
# ダブルクリックで動画AIツールを起動し、ブラウザを開きます。
cd "$(dirname "$0")" || exit 1

URL="http://127.0.0.1:5001"

# 「起動済み」判定は HTTP 200 のときだけ（403等は起動扱いにしない）
is_up() { [ "$(curl -s -o /dev/null -w '%{http_code}' "$URL" 2>/dev/null)" = "200" ]; }

if is_up; then
  echo "✅ すでに起動しています。ブラウザを開きます…"
else
  echo "🚀 動画AIツールを起動しています…"
  nohup /usr/bin/python3 app.py > server.log 2>&1 &
  for i in $(seq 1 25); do
    if is_up; then break; fi
    sleep 1
  done
fi

if is_up; then
  open "$URL"
  echo "🎬 ブラウザで $URL を開きました。動画をアップロードしてください。"
  echo "（このウィンドウは閉じてOKです。ツールはバックグラウンドで動き続けます）"
else
  echo "⚠️ 起動に失敗しました。server.log を確認してください："
  tail -15 server.log
  read -r -p "Enterで閉じます…" _
fi
