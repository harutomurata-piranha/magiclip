"""
環境設定。APP_ENV で .env.development / .env.production を切り替える。

- development: Stripeテストモード、本数制限なし
- production : Stripe本番、本数制限あり
"""
import os
from dotenv import load_dotenv

APP_ENV = os.getenv("APP_ENV", "development")

# 環境ごとの .env を読み込む（無ければ既存の .env をフォールバックで読む）
_env_file = f".env.{APP_ENV}"
if os.path.exists(_env_file):
    load_dotenv(_env_file)
else:
    load_dotenv()  # フォールバック


def _bool(name, default=False):
    return os.getenv(name, str(default)).lower() in ("1", "true", "yes", "on")


class Config:
    ENV = APP_ENV
    IS_DEV = APP_ENV == "development"

    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-me")
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL", "sqlite:///magiclip.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # アップロード/出力
    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
    OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER", "outputs")
    MAX_CONTENT_MB = int(os.getenv("MAX_CONTENT_MB", "500"))

    # 本数制限（開発環境では無制限）
    NO_LIMIT = _bool("NO_LIMIT", default=IS_DEV)
    FREE_TRIAL_CREDITS = int(os.getenv("FREE_TRIAL_CREDITS", "3"))

    # 限定公開：ACCESS_PASSWORD を設定すると、サイト全体にパスワード（Basic認証）がかかる。
    # 未設定なら誰でも閲覧可（ローカル開発はこちら）。知人限定デモ等はこれで“鍵”をかける。
    ACCESS_USER = os.getenv("ACCESS_USER", "guest")
    ACCESS_PASSWORD = os.getenv("ACCESS_PASSWORD", "")

    # 料金（円）
    PRICE_PAYG_JPY = int(os.getenv("PRICE_PAYG_JPY", "100"))      # 従量課金：1本100円
    PRICE_MONTHLY_JPY = int(os.getenv("PRICE_MONTHLY_JPY", "1980"))  # 月額プラン

    # Stripe（開発はテストキー）
    STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
    STRIPE_PUBLISHABLE_KEY = os.getenv("STRIPE_PUBLISHABLE_KEY", "")
    STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")

    @property
    def stripe_enabled(self):
        # テストキー含め、シークレットキーがあれば決済を有効化。無ければ開発用に決済をシミュレート
        return bool(self.STRIPE_SECRET_KEY)


config = Config()
