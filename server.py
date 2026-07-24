"""
MagiClip ビジネスアプリ本体（「箱」）。

AI編集エンジンは video_processor.process_video() に分離してあり、後から差し替え可能。
画面: LP / 登録・ログイン / プラン選択 / Stripe決済 / マイpage / アップロード / 処理中 / 完成
"""
import os
import json
import uuid
import threading

import app as engine          # AI編集エンジン（ヘルスチェックでクライアントを使う）
import stripe
from flask import (Flask, render_template, request, redirect, url_for, flash,
                   jsonify, send_file, abort)
from flask_login import (LoginManager, login_user, logout_user, login_required,
                         current_user)
from werkzeug.utils import secure_filename

from config import config
from models import db, User, Job, Payment, PLAN_PAYG, PLAN_MONTHLY
from video_processor import (process_video, load_edit_data,
                             list_segments, reedit_segments, segment_thumbnail)


def _friendly_error(e):
    """例外を、ユーザーに見せる分かりやすい日本語に変換する。"""
    s = str(e).lower()
    if "credit balance" in s or "billing" in s:
        return "AIの利用クレジットが不足しています。少し時間をおくか、管理者にご連絡ください。"
    if "rate_limit" in s or "overloaded" in s or "429" in s or "529" in s:
        return "AIが混み合っています。少し時間をおいて、もう一度お試しください。"
    if "authentication" in s or "api key" in s or "401" in s:
        return "AIの認証に失敗しました。設定をご確認ください（管理者向け）。"
    if "編集構成を作れません" in str(e):
        return "この動画から編集を作れませんでした（短すぎる/無音、またはAIの一時的な不調の可能性）。別の動画でお試しください。"
    return "編集に失敗しました。もう一度お試しください。"


def create_app():
    app = Flask(__name__)
    app.config.from_object(config)
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_MB * 1024 * 1024

    os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(config.OUTPUT_FOLDER, exist_ok=True)

    db.init_app(app)
    with app.app_context():
        db.create_all()

    login_manager = LoginManager(app)
    login_manager.login_view = "login"
    login_manager.login_message = "ログインしてください。"

    @login_manager.user_loader
    def load_user(user_id):
        return db.session.get(User, int(user_id))

    if config.stripe_enabled:
        stripe.api_key = config.STRIPE_SECRET_KEY

    # 全テンプレに渡す共通情報
    @app.context_processor
    def inject_globals():
        return {"cfg": config}

    # ---------------- ランディング ----------------
    @app.route("/")
    def index():
        return render_template("index.html")

    # ---------------- ヘルスチェック（AIが使える状態かを確認）----------------
    @app.route("/health")
    def health():
        try:
            engine.anthropic_client.messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=1,
                messages=[{"role": "user", "content": "ping"}])
            return jsonify({"ok": True, "detail": "AI接続OK（クレジット有効）"})
        except Exception as e:
            return jsonify({"ok": False, "detail": _friendly_error(e)})

    # ---------------- 認証 ----------------
    @app.route("/register", methods=["GET", "POST"])
    def register():
        if current_user.is_authenticated:
            return redirect(url_for("mypage"))
        if request.method == "POST":
            email = (request.form.get("email") or "").strip().lower()
            pw = request.form.get("password") or ""
            if not email or not pw:
                flash("メールアドレスとパスワードを入力してください。", "error")
            elif User.query.filter_by(email=email).first():
                flash("このメールアドレスは登録済みです。", "error")
            else:
                u = User(email=email, plan="free", credits=config.FREE_TRIAL_CREDITS)
                u.set_password(pw)
                db.session.add(u)
                db.session.commit()
                login_user(u)
                flash(f"登録完了！お試しで{config.FREE_TRIAL_CREDITS}本まで無料です。", "ok")
                return redirect(url_for("mypage"))
        return render_template("register.html")

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if current_user.is_authenticated:
            return redirect(url_for("mypage"))
        if request.method == "POST":
            email = (request.form.get("email") or "").strip().lower()
            pw = request.form.get("password") or ""
            u = User.query.filter_by(email=email).first()
            if u and u.check_password(pw):
                login_user(u)
                return redirect(url_for("mypage"))
            flash("メールアドレスまたはパスワードが違います。", "error")
        return render_template("login.html")

    @app.route("/logout")
    @login_required
    def logout():
        logout_user()
        return redirect(url_for("index"))

    # ---------------- プラン選択 ----------------
    @app.route("/plans")
    def plans():
        return render_template("plans.html")

    # ---------------- Stripe決済 ----------------
    def _fulfill(session_id, user_id, kind):
        """決済を反映（冪等：同じセッションは一度だけ）"""
        if Payment.query.filter_by(stripe_session_id=session_id).first():
            return
        user = db.session.get(User, int(user_id))
        if not user:
            return
        if kind == "payg":
            user.credits += 1
            user.plan = PLAN_PAYG
        elif kind == "monthly":
            user.subscription_active = True
            user.plan = PLAN_MONTHLY
        db.session.add(Payment(user_id=user.id, stripe_session_id=session_id, kind=kind))
        db.session.commit()

    @app.route("/checkout/<kind>", methods=["POST"])
    @login_required
    def checkout(kind):
        if kind not in ("payg", "monthly"):
            abort(404)

        # 開発で Stripe 未設定なら決済をシミュレート（テストしやすく）
        if not config.stripe_enabled:
            sim_id = f"sim_{kind}_{uuid.uuid4().hex[:8]}"
            _fulfill(sim_id, current_user.id, kind)
            flash("（開発モード）決済をシミュレートしました。", "ok")
            return redirect(url_for("mypage"))

        base = request.url_root.rstrip("/")
        success = base + url_for("checkout_success") + "?session_id={CHECKOUT_SESSION_ID}"
        cancel = base + url_for("plans")
        common = dict(
            success_url=success, cancel_url=cancel,
            client_reference_id=str(current_user.id),
            metadata={"user_id": str(current_user.id), "kind": kind},
        )
        if kind == "payg":
            session = stripe.checkout.Session.create(
                mode="payment",
                line_items=[{
                    "price_data": {
                        "currency": "jpy",
                        "unit_amount": config.PRICE_PAYG_JPY,
                        "product_data": {"name": "MagiClip 1本（従量課金）"},
                    },
                    "quantity": 1,
                }],
                **common,
            )
        else:
            session = stripe.checkout.Session.create(
                mode="subscription",
                line_items=[{
                    "price_data": {
                        "currency": "jpy",
                        "unit_amount": config.PRICE_MONTHLY_JPY,
                        "recurring": {"interval": "month"},
                        "product_data": {"name": "MagiClip 月額プラン"},
                    },
                    "quantity": 1,
                }],
                **common,
            )
        return redirect(session.url, code=303)

    @app.route("/checkout/success")
    @login_required
    def checkout_success():
        session_id = request.args.get("session_id")
        if session_id and config.stripe_enabled:
            try:
                s = stripe.checkout.Session.retrieve(session_id)
                if s.get("payment_status") in ("paid", "no_payment_required") or s.get("status") == "complete":
                    md = s.get("metadata") or {}
                    _fulfill(session_id, md.get("user_id", current_user.id), md.get("kind", "payg"))
            except Exception:
                pass
        flash("ご購入ありがとうございます！", "ok")
        return redirect(url_for("mypage"))

    @app.route("/webhook/stripe", methods=["POST"])
    def stripe_webhook():
        payload = request.data
        sig = request.headers.get("Stripe-Signature", "")
        try:
            if config.STRIPE_WEBHOOK_SECRET:
                event = stripe.Webhook.construct_event(payload, sig, config.STRIPE_WEBHOOK_SECRET)
            else:
                event = stripe.Event.construct_from(request.get_json(force=True), stripe.api_key)
        except Exception:
            return "bad signature", 400

        if event["type"] == "checkout.session.completed":
            s = event["data"]["object"]
            md = s.get("metadata") or {}
            if md.get("user_id"):
                _fulfill(s["id"], md["user_id"], md.get("kind", "payg"))
        elif event["type"] == "customer.subscription.deleted":
            sub = event["data"]["object"]
            cust = sub.get("customer")
            u = User.query.filter_by(stripe_customer_id=cust).first()
            if u:
                u.subscription_active = False
                db.session.commit()
        return "", 200

    # ---------------- マイページ ----------------
    @app.route("/mypage")
    @login_required
    def mypage():
        jobs = Job.query.filter_by(user_id=current_user.id).order_by(Job.created_at.desc()).limit(20).all()
        return render_template("mypage.html", jobs=jobs)

    # ---------------- アップロード ----------------
    @app.route("/upload", methods=["GET", "POST"])
    @login_required
    def upload():
        if request.method == "POST":
            ok, reason = current_user.can_process(no_limit=config.NO_LIMIT)
            if not ok:
                flash(reason, "error")
                return redirect(url_for("plans"))

            file = request.files.get("video")
            if not file or not file.filename:
                flash("動画ファイルを選んでください。", "error")
                return redirect(url_for("upload"))

            job_id = uuid.uuid4().hex
            safe = secure_filename(file.filename) or "input.mp4"
            in_path = os.path.join(config.UPLOAD_FOLDER, f"{job_id}_{safe}")
            out_path = os.path.join(config.OUTPUT_FOLDER, f"{job_id}_output.mp4")
            file.save(in_path)

            mode = "pro" if request.form.get("mode") == "pro" else "auto"
            job = Job(id=job_id, user_id=current_user.id, filename=safe, status="待機中", mode=mode)
            db.session.add(job)
            current_user.consume_credit(no_limit=config.NO_LIMIT)  # 1本消費（開発/月額は減らない）
            db.session.commit()

            t = threading.Thread(
                target=_run_job,
                args=(app, job_id, in_path, out_path, current_user.plan),
                daemon=True,
            )
            t.start()
            return redirect(url_for("processing", job_id=job_id))
        return render_template("upload.html")

    def _run_job(flask_app, job_id, in_path, out_path, plan):
        with flask_app.app_context():
            job = db.session.get(Job, job_id)
            job.status = "処理中"
            job.error = None
            db.session.commit()
            try:
                process_video(in_path, out_path, plan)
                job.status = "完成"
                job.output_path = out_path
            except Exception as e:
                job.status = "エラー"
                job.error = _friendly_error(e)
                _refund_credit(job)   # 失敗したら消費した1本を返す
            db.session.commit()

    # ---------------- 処理中 / 状態 ----------------
    @app.route("/processing/<job_id>")
    @login_required
    def processing(job_id):
        job = _owned_job(job_id)
        return render_template("processing.html", job=job)

    def _refund_credit(job):
        """失敗時、アップロードで消費した1本を返す（無制限/月額は元々消費なし）。"""
        if config.NO_LIMIT:
            return
        user = db.session.get(User, job.user_id)
        if user and not user.subscription_active:
            user.credits += 1

    @app.route("/status/<job_id>")
    @login_required
    def status(job_id):
        job = _owned_job(job_id)
        return jsonify({"status": job.status, "error": job.error})

    # ---------------- 完成（プレビュー / ダウンロード）----------------
    @app.route("/result/<job_id>")
    @login_required
    def result(job_id):
        job = _owned_job(job_id)
        if job.status != "完成":
            return redirect(url_for("processing", job_id=job_id))
        return render_template("result.html", job=job)

    @app.route("/video/<job_id>")
    @login_required
    def video(job_id):
        job = _owned_job(job_id)
        if not job.output_path or not os.path.exists(job.output_path):
            abort(404)
        return send_file(job.output_path, mimetype="video/mp4")

    @app.route("/download/<job_id>")
    @login_required
    def download(job_id):
        job = _owned_job(job_id)
        if not job.output_path or not os.path.exists(job.output_path):
            abort(404)
        return send_file(job.output_path, as_attachment=True,
                         download_name=f"magiclip_{job_id}.mp4")

    # ---------------- プロ編集（AIの編集をシーン一覧で修正：サムネ＋字幕）----------------
    @app.route("/edit/<job_id>", methods=["GET", "POST"])
    @login_required
    def edit(job_id):
        job = _owned_job(job_id)
        scenes = list_segments(job.output_path) if job.output_path else []
        if not scenes:
            flash("この動画には編集データがありません。", "error")
            return redirect(url_for("result", job_id=job_id))

        if request.method == "POST":
            # 各シーンの編集結果(JSON)：選択(keep)＋編集テキスト。選んだシーンの映像だけが残る。
            try:
                edited = json.loads(request.form.get("scenes_json", "[]"))
            except ValueError:
                edited = []
            by_i = {s["i"]: s for s in scenes}
            kept = []
            for s in edited:
                if s.get("keep") and s.get("i") in by_i and (s.get("text") or "").strip():
                    src = by_i[s["i"]]
                    kept.append({"i": s["i"], "text": s.get("text", ""),
                                 "start": src["start"], "end": src["end"]})
            if not kept:
                flash("少なくとも1つのシーンを残してください（チェック＆文字あり）。", "error")
                return redirect(url_for("edit", job_id=job_id))
            font_key = request.form.get("font", "auto")          # 書体の手動指定（auto=AIにおまかせ）
            job.status = "処理中"
            db.session.commit()
            threading.Thread(target=_run_reedit,
                             args=(app, job_id, job.output_path, kept, font_key),
                             daemon=True).start()
            return redirect(url_for("processing", job_id=job_id))

        data = load_edit_data(job.output_path) or {}
        return render_template("edit.html", job=job, scenes=scenes,
                               font_opts=engine.font_options(),
                               cur_font=data.get("font_key", "auto"),
                               ai_font=engine.pick_subtitle_font(data.get("bgm_mood", ""))[2])

    @app.route("/thumb/<job_id>/<int:idx>")
    @login_required
    def thumb(job_id, idx):
        job = _owned_job(job_id)
        path = segment_thumbnail(job.output_path, idx) if job.output_path else None
        if not path or not os.path.exists(path):
            abort(404)
        return send_file(path, mimetype="image/jpeg")

    def _run_reedit(flask_app, job_id, output_path, kept_scenes, font_key=None):
        with flask_app.app_context():
            job = db.session.get(Job, job_id)
            job.status = "処理中"
            job.error = None
            db.session.commit()
            try:
                reedit_segments(output_path, kept_scenes, font_key)
                job.status = "完成"
            except Exception as e:
                job.status = "エラー"
                job.error = _friendly_error(e)
            db.session.commit()

    def _owned_job(job_id):
        job = db.session.get(Job, job_id)
        if not job or job.user_id != current_user.id:
            abort(404)
        return job

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=config.IS_DEV)
