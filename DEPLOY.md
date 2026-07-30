# MagiClip デプロイ手順（限定公開・スマホから試せるようにする）

目的：PCを起動していなくても、出先のスマホから **パスワード付き** でMagiClipを試せるようにする。
方式：Docker イメージをマネージドPaaSに載せる（例は **Render**。Railway / Fly.io でもほぼ同じ）。

---

## 前提（用意するもの）
- GitHub アカウント（このリポジトリをpushしておく）
- OpenAI / Anthropic の APIキー
- クレジットカード（PaaSの登録に必要な場合あり。無料枠でも試せる）

> ⚠️ 秘密情報（APIキー等）は**コードにも .env.production にも入れず**、PaaSのダッシュボードの「環境変数」に設定します。`.env*` は `.dockerignore` で除外済みです。

---

## 手順（Render の例）

### 1. リポジトリをGitHubにpush
```bash
git add -A && git commit -m "本番デプロイ用: Dockerfile + 限定公開ゲート"
# git push は自分の手で（このリポジトリの権限設定で push は私からは実行できません）
git push
```

### 2. Render で Web Service を作成
1. https://render.com にログイン →「New +」→「Web Service」
2. このGitHubリポジトリを接続
3. **Runtime: Docker**（リポジトリ直下の `Dockerfile` が自動で使われる）
4. Instance Type: まずは有料の最小(Starter)推奨（無料枠はスリープ＆メモリが小さく、動画処理が不安定なことがある）

### 3. 環境変数を設定（Environment タブ）
`.env.production.example` を参考に、**実際の値**を入れる：

| キー | 値 | 必須 |
|---|---|---|
| `APP_ENV` | `production` | ✅ |
| `SECRET_KEY` | 長いランダム文字列 | ✅ |
| `OPENAI_API_KEY` | あなたのOpenAI鍵 | ✅ |
| `ANTHROPIC_API_KEY` | あなたのAnthropic鍵 | ✅ |
| `ACCESS_USER` | 例: `guest` | 限定公開に✅ |
| `ACCESS_PASSWORD` | 知人と共有する合言葉 | 限定公開に✅ |
| `NO_LIMIT` | `true`（デモ中は本数無制限） | 任意 |

> `SECRET_KEY` の作り方: `python3 -c "import secrets;print(secrets.token_hex(32))"`

### 4. デプロイ
「Create Web Service」→ 自動ビルド＆起動。`https://xxxx.onrender.com` のURLが発行される。

### 5. スマホから確認
- スマホのブラウザでそのURLを開く
- **ユーザー名 `guest` ＋ 設定したパスワード** を入力すると入れる（＝限定公開）
- そのままアップロード → 生成、を試せる（PCは不要）

---

## 動作の要点・注意（デモ用として割り切る点）
- **ストレージは揮発性**：再デプロイ/再起動で、生成した動画・登録ユーザー・クレジットはリセットされる（デモなら問題なし。継続運用するなら永続ディスク＋外部DBが必要）。
- **処理性能**：動画処理はCPUを使う。短い動画ほど軽快。長尺・高解像度は遅い/失敗することがある → 小さいインスタンスなら短めの動画で。
- **アップロード上限**：`MAX_CONTENT_MB`（既定500MB）。PaaS側のリクエスト上限にも注意。
- **課金(Stripe)**：デモは無料クレジットで回せるので未設定でOK。使う場合のみ `STRIPE_*` を設定＋Stripe側でwebhook(`/webhook/stripe`)を登録。
- **鍵の安全**：APIキーはPaaSの環境変数だけに置く。GitHubには絶対pushしない（`.dockerignore`/`.gitignore`で除外済み）。

---

## 限定公開の鍵を変える/外す
- パスワードを変える：`ACCESS_PASSWORD` を変更して再デプロイ
- 公開（誰でも可）にする：`ACCESS_PASSWORD` を空にする（＝ゲート無効）

## ローカルで限定公開を試す
```bash
ACCESS_PASSWORD=test1234 PORT=5050 python3 server.py
# ブラウザで http://127.0.0.1:5050 → guest / test1234 で入れる
```
