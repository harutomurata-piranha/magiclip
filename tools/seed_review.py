"""UIレビュー用の環境をつくる（開発DB専用。本番コードには影響しない）。

使い方:
    python3 tools/seed_review.py            # レビュー用アカウントとサンプルを投入
    python3 tools/seed_review.py --empty    # 空状態のアカウントだけにする（履歴0件）

投入するもの:
  - review@magiclip.test  / review1234  … 一般プラン。動画が数件ある状態
  - pro@magiclip.test     / review1234  … 月額プラン（有料）。同上
  - empty@magiclip.test   / review1234  … 動画0件（空状態の確認用）
  - 状態確認用のジョブ（処理中 / エラー各種）
"""
import os
import sys
import uuid
from datetime import datetime, timedelta

os.environ.setdefault("APP_ENV", "development")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import create_app                      # noqa: E402
from models import db, User, Job                   # noqa: E402

PASSWORD = "review1234"
ACCOUNTS = [
    ("review@magiclip.test", "free", 3, False),
    ("pro@magiclip.test", "monthly", 0, True),
    ("empty@magiclip.test", "free", 3, False),
]

# 長短を混ぜたファイル名（表示崩れの確認用）
SAMPLES = [
    ("2026-07-30_渋谷スクランブル交差点_インタビュー_ロングバージョン_最終版_v3.MOV", "完成", "pro"),
    ("IMG_0798.MOV", "完成", "pro"),
    ("a.mp4", "完成", "auto"),
    ("商品紹介_カメラレビュー.mov", "完成", "auto"),
]


def upsert_user(email, plan, credits, sub):
    u = User.query.filter_by(email=email).first()
    if not u:
        u = User(email=email)
        db.session.add(u)
    u.plan, u.credits, u.subscription_active = plan, credits, sub
    u.set_password(PASSWORD)
    db.session.commit()
    return u


def add_job(user, filename, status, mode, minutes_ago, output=None,
            progress=0, step=None, eta=None, error=None):
    j = Job(id=uuid.uuid4().hex, user_id=user.id, filename=filename,
            status=status, mode=mode, output_path=output,
            progress=progress, step=step, eta_sec=eta, error=error,
            created_at=datetime.utcnow() - timedelta(minutes=minutes_ago))
    db.session.add(j)
    db.session.commit()
    return j


def main():
    empty_only = "--empty" in sys.argv
    app = create_app()
    with app.app_context():
        # 既存の完成ジョブ（実データ）を流用して、動画が再生できる状態にする
        done = Job.query.filter_by(status="完成").filter(Job.output_path.isnot(None)).all()
        real = [j for j in done if j.output_path and os.path.exists(j.output_path)]
        print(f"再生可能な既存の完成ジョブ: {len(real)}件")

        users = {}
        for email, plan, credits, sub in ACCOUNTS:
            u = upsert_user(email, plan, credits, sub)
            users[email] = u
            Job.query.filter_by(user_id=u.id).delete()      # 毎回作り直す
            db.session.commit()
            print(f"アカウント: {email} / {PASSWORD}  (plan={plan}, sub={sub})")

        if empty_only:
            print("空状態のみ用意しました（全アカウントの履歴を0件に）")
            return

        for email in ("review@magiclip.test", "pro@magiclip.test"):
            u = users[email]
            for i, (name, status, mode) in enumerate(SAMPLES):
                out = real[i % len(real)].output_path if real else None
                add_job(u, name, status, mode, minutes_ago=30 + i * 47, output=out, progress=100)
            # 状態確認用
            add_job(u, "処理中サンプル.mov", "処理中", "auto", 5,
                    progress=62, step="テンポよくカットしています", eta=118)
            add_job(u, "失敗サンプル.mov", "エラー", "auto", 12,
                    error="この動画から編集を作れませんでした（短すぎる/無音、またはAIの一時的な不調の可能性）。別の動画でお試しください。")
            print(f"  {email}: 動画 {Job.query.filter_by(user_id=u.id).count()} 件")

        print("\n完了。レビュー用URLは README を参照。")


if __name__ == "__main__":
    main()
