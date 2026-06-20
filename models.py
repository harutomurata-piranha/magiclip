"""
データモデル（User / Job / Payment）。
"""
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

# プラン定義
PLAN_FREE = "free"        # お試し無料（FREE_TRIAL_CREDITS 本まで）
PLAN_PAYG = "payg"        # 従量課金（1本ずつ購入）
PLAN_MONTHLY = "monthly"  # 月額プラン（期間中は無制限）

PLAN_LABELS = {
    PLAN_FREE: "お試し無料",
    PLAN_PAYG: "従量課金",
    PLAN_MONTHLY: "月額プラン",
}


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)

    plan = db.Column(db.String(20), default=PLAN_FREE, nullable=False)
    credits = db.Column(db.Integer, default=0, nullable=False)       # 残り本数（free/payg用）
    subscription_active = db.Column(db.Boolean, default=False)        # 月額が有効か
    stripe_customer_id = db.Column(db.String(255))

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    jobs = db.relationship("Job", backref="user", lazy=True)

    def set_password(self, pw):
        # pbkdf2:sha256 を明示（環境によっては既定の scrypt が使えないため）
        self.password_hash = generate_password_hash(pw, method="pbkdf2:sha256")

    def check_password(self, pw):
        return check_password_hash(self.password_hash, pw)

    def can_process(self, no_limit=False):
        """この動画を処理できるか（できない理由も返す）"""
        if no_limit:
            return True, ""
        if self.subscription_active:
            return True, ""
        if self.credits > 0:
            return True, ""
        return False, "残り本数がありません。プランを購入してください。"

    def consume_credit(self, no_limit=False):
        """1本消費する（無制限・月額有効中は減らさない）"""
        if no_limit or self.subscription_active:
            return
        if self.credits > 0:
            self.credits -= 1

    @property
    def plan_label(self):
        if self.subscription_active:
            return PLAN_LABELS[PLAN_MONTHLY]
        return PLAN_LABELS.get(self.plan, self.plan)


class Job(db.Model):
    id = db.Column(db.String(36), primary_key=True)            # uuid
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    filename = db.Column(db.String(255))
    status = db.Column(db.String(20), default="待機中")        # 待機中/処理中/完成/エラー
    mode = db.Column(db.String(10), default="auto")            # auto=一般(全自動) / pro=プロ(編集可)
    output_path = db.Column(db.String(500))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Payment(db.Model):
    """決済の冪等記録（同じStripeセッションを二重に反映しないため）"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    stripe_session_id = db.Column(db.String(255), unique=True, nullable=False)
    kind = db.Column(db.String(20))                            # payg / monthly
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
