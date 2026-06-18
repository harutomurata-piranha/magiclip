#!/bin/bash
# ダブルクリックで動画AIツールを停止します。
cd "$(dirname "$0")"
pkill -f "python3 app.py" 2>/dev/null && echo "🛑 動画AIツールを停止しました。" || echo "（すでに停止しています）"
sleep 1
