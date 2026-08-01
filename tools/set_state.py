"""UIレビュー用に、ジョブの状態をDBで直接書き換える（開発DB専用）。

進捗や失敗の見え方を、実処理を待たずに落ち着いて確認するためのもの。
アプリ側にレビュー用フラグを足していないので、本番には一切影響しない。

使い方（job_id はマイページのURLか下のlistで確認）:

    python3 tools/set_state.py list
    python3 tools/set_state.py <job_id> progress 62         # 62%で固定（工程名と残りも自動で妥当な値に）
    python3 tools/set_state.py <job_id> progress 0
    python3 tools/set_state.py <job_id> error credit        # エラー: クレジット不足
    python3 tools/set_state.py <job_id> error rate          # エラー: 混雑
    python3 tools/set_state.py <job_id> error auth          # エラー: 認証
    python3 tools/set_state.py <job_id> error material      # エラー: 素材が不適
    python3 tools/set_state.py <job_id> error generic       # エラー: 汎用
    python3 tools/set_state.py <job_id> done                # 完成に戻す
"""
import os
import sys

os.environ.setdefault("APP_ENV", "development")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import create_app          # noqa: E402
from models import db, Job             # noqa: E402

# 進捗％ → 画面に出る工程名（video_processor.STEP_WEIGHTS と対応）
STEPS = [(0, "準備しています"), (5, "音声を取り出しています"),
         (30, "話している内容を聞き取っています"), (50, "どこを残すか考えています"),
         (75, "テンポよくカットしています"), (85, "字幕をつけています"),
         (100, "仕上げています（BGM・書体）")]

ERRORS = {
    "credit":   "AIの利用クレジットが不足しています。少し時間をおくか、管理者にご連絡ください。",
    "rate":     "AIが混み合っています。少し時間をおいて、もう一度お試しください。",
    "auth":     "AIの認証に失敗しました。設定をご確認ください（管理者向け）。",
    "material": "この動画から編集を作れませんでした（短すぎる/無音、またはAIの一時的な不調の可能性）。別の動画でお試しください。",
    "generic":  "編集に失敗しました。もう一度お試しください。",
}


def step_for(p):
    label = STEPS[0][1]
    for th, name in STEPS:
        if p >= th:
            label = name
    return label


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    app = create_app()
    with app.app_context():
        if sys.argv[1] == "list":
            for j in Job.query.order_by(Job.created_at.desc()).limit(20):
                print(f"{j.id}  {j.status:5} {str(j.progress or 0):>3}%  {j.filename}")
            return

        job = db.session.get(Job, sys.argv[1])
        if not job:
            print("job が見つかりません:", sys.argv[1])
            return
        cmd = sys.argv[2] if len(sys.argv) > 2 else "done"

        if cmd == "progress":
            p = int(sys.argv[3]) if len(sys.argv) > 3 else 62
            job.status, job.progress = "処理中", p
            job.step = step_for(p)
            job.eta_sec = max(5, int(180 * (100 - p) / 100))
            job.error = None
        elif cmd == "error":
            kind = sys.argv[3] if len(sys.argv) > 3 else "material"
            job.status, job.error = "エラー", ERRORS.get(kind, ERRORS["generic"])
        elif cmd == "done":
            job.status, job.progress, job.step, job.eta_sec, job.error = "完成", 100, "完成しました", 0, None
        else:
            print(__doc__)
            return
        db.session.commit()
        print(f"{job.id} → status={job.status} progress={job.progress} step={job.step} "
              f"eta={job.eta_sec} error={(job.error or '')[:40]}")


if __name__ == "__main__":
    main()
