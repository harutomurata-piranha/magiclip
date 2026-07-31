from flask import Flask, request, jsonify, send_file, render_template_string
import os
import threading
import uuid
import subprocess
import json
import re
import difflib
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import shutil
import openai
import anthropic
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

openai_client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

jobs = {}

# 同梱フォント置き場（OFL＝商用利用可の日本語フォント。OSに依存せずどこでも同じ見た目にする）
FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")


def _resolve_font():
    """字幕用の日本語フォントを探す。同梱フォント最優先→OS標準（mac=ヒラギノ / Linux=Noto CJK）"""
    import glob
    candidates = [
        os.environ.get("SUBTITLE_FONT", ""),
        f"{FONT_DIR}/NotoSansJP[wght].ttf",                  # 同梱（商用可・環境非依存）
        "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",   # macOS
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",   # Linux (fonts-noto-cjk)
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    for pat in ("/usr/share/fonts/**/NotoSansCJK*.*", "/usr/share/fonts/**/NotoSerifCJK*.*"):
        hit = glob.glob(pat, recursive=True)
        if hit:
            return hit[0]
    return None

SUBTITLE_FONT = _resolve_font()


def load_font(path, size, index=0, variation=None):
    """フォント読み込み。可変フォント(Noto等)はウェイトを必ず指定する
    ——未指定だと極細で描画され、動画上でほぼ読めなくなるため既定はBold。"""
    font = ImageFont.truetype(path, size, index=index)
    try:
        names = [n.decode() if isinstance(n, bytes) else n for n in font.get_variation_names()]
        want = variation if variation in names else ("Bold" if "Bold" in names else None)
        if want:
            font.set_variation_by_name(want)
    except Exception:
        pass          # 可変でないフォントはそのまま
    return font

def get_video_duration(video_path):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path], capture_output=True, text=True)
    return float(result.stdout.strip())

def extract_audio(video_path, audio_path):
    subprocess.run(["ffmpeg", "-i", video_path, "-vn", "-ar", "44100", "-ac", "2", "-b:a", "128k", audio_path, "-y"], capture_output=True)

def transcribe_audio(audio_path):
    audio_file = open(audio_path, "rb")
    return openai_client.audio.transcriptions.create(
        model="whisper-1", file=audio_file, language="ja",
        response_format="verbose_json", timestamp_granularities=["segment", "word"])

# 聞き取りの信頼度フィルタ（聞き取れるシーンだけで構成するため）
MIN_AVG_LOGPROB = -1.0      # これ未満は認識の信頼度が低い → 除外
MAX_NO_SPEECH_PROB = 0.6    # これ以上は無音・雑音の可能性が高い → 除外
FILLER_TOKENS = ["えーと", "えっと", "ええと", "あの", "その", "うーん",
                 "はい", "うん", "ええ", "あー", "えー", "んー", "まあ"]

def is_clear_segment(segment):
    """聞き取れる（信頼度が高く、相槌・フィラーだけでない）セグメントか判定する"""
    if getattr(segment, "no_speech_prob", 0.0) >= MAX_NO_SPEECH_PROB:
        return False
    if getattr(segment, "avg_logprob", 0.0) <= MIN_AVG_LOGPROB:
        return False
    stripped = segment.text.strip()
    for t in FILLER_TOKENS:
        stripped = stripped.replace(t, "")
    stripped = re.sub(r"[、。「」・,.!?！？\s]", "", stripped)
    return len(stripped) >= 2

def clean_segments(transcript):
    cleaned = []
    for segment in transcript.segments:
        if not is_clear_segment(segment):
            continue
        text = segment.text.strip()
        duration = segment.end - segment.start
        if duration < 1.0 and len(text) < 5:
            continue
        if duration > 10:
            parts = re.split(r'([、。])', text)
            merged = []
            current = ""
            for part in parts:
                current += part
                if part in ['。', '、'] and len(current) > 10:
                    merged.append(current.strip())
                    current = ""
            if current.strip():
                merged.append(current.strip())
            if len(merged) > 1:
                part_duration = duration / len(merged)
                lp = getattr(segment, "avg_logprob", 0.0)
                for j, part_text in enumerate(merged):
                    cleaned.append({"start": segment.start + j * part_duration, "end": segment.start + (j + 1) * part_duration, "text": part_text, "avg_logprob": lp})
                continue
        cleaned.append({"start": segment.start, "end": segment.end, "text": text, "avg_logprob": getattr(segment, "avg_logprob", 0.0)})
    return cleaned

# 編集判断のしきい値（シンプルなルールベース編集）
GAP_FLAG_SEC = 1.0       # これ以上の無音/間は「間が空いている」サイン → 余白・文脈切れの候補
UNCLEAR_LOGPROB = -0.6   # これ未満は聞き取りが怪しい（言い間違い・崩れ・不明瞭の可能性）

# 「もう一度生成」時の微調整（編集哲学は変えず、削り具合だけ変えて違いを出す）
REGEN_NOTES = [
    "",  # 1本目：標準
    "前回より思い切って削る：間延び・冗長・つながりの弱い所をさらにカットし、もっとテンポを上げる。",
    "前回より少し残す：文脈が分かることを優先し、つながりに必要な所は無理に削らない。",
    "取捨選択を前回と変える：残すか迷う境界のセグメントの採否を前回と入れ替え、別パターンに見せる（哲学は同じ）。",
]

def _annotate_segments(cleaned_segments):
    """各セグメントに「前との間(gap)」「聞き取りの怪しさ」を注記し、編集判断の材料にする。"""
    lines = []
    for i, s in enumerate(cleaned_segments):
        flags = []
        if i > 0:
            gap = s["start"] - cleaned_segments[i - 1]["end"]
            if gap >= GAP_FLAG_SEC:
                flags.append(f"前と{gap:.1f}秒の間")
        if s.get("avg_logprob", 0.0) <= UNCLEAR_LOGPROB:
            flags.append("聞き取りが怪しい")
        note = f"（{' / '.join(flags)}）" if flags else ""
        lines.append(f"[{s['start']:.1f}秒〜{s['end']:.1f}秒]{note} {s['text']}")
    return "\n".join(lines)

JSON_SPEC = '{"bgm_mood": "動画に合うBGMのムード", "scenes": [{"start": 開始秒, "end": 終了秒, "reason": "選んだ理由", "transition": "cut または fade"}]}'

# 構成（シーン判定）・分析は Sonnet で文脈理解（コストと品質のバランス）
STRUCTURE_MODEL = "claude-sonnet-4-6"
# 字幕の校正・簡潔化に使うモデル（軽量・低コスト）
SUBTITLE_MODEL = "claude-haiku-4-5-20251001"

def _shared_rules(duration):
    return f"""【MagiClipの編集方針】話している内容は、撮影者が「伝えたいこと」。だから発話は原則すべて残し、
無駄だけを取り除いてテンポを上げる。良い所を"選び抜いて絞る"のではなく、不要な所だけを"落とす"——これが基本姿勢。
話しているシーンを丸ごとカットすると「伝えたかったことが消えた」と不信につながる。迷ったら残す。

【原則すべて残す】
- 何かを話しているシーンは原則すべて残す。地味でも・短くても、伝えたい内容が入っている
- お店・地名・施設の紹介、感想、注意、豆知識などの具体的な情報は必ず残す
- 言い淀みや小さな間も、人柄が伝わるなら無理に消さない

【"落として良い"のは次のものだけ】
- 日本語として意味が通らない：音声認識が崩れて文にならない／言い間違いで意味不明な箇所
- 長い無音・言葉のない間（"○秒の間"の注記）
- 内容がまったく無い、フィラー（えー・あの等）だけの箇所

【判断の物差し（重要）】
- 「重要か・面白いか・価値があるか」では判断しない。判断基準は『日本語の文として成立しているか』だけ
- 短い感想（例：いい店です／安い／美味しい）や補足情報（例：喫煙できます）も、話者が伝えたい内容。地味でも残す
- 前後で1つの話になっている場合、片方だけ残して片方を落とさない（お店の名前とその感想はセットで残す）

【作り方】
- テンポは「無音・崩れ・意味不明・重複」を取り除くことで作る。意味の通る発話は削って短くしない
- transition は基本"cut"（テンポ重視）。話題が大きく変わる節目だけ"fade"

【シーンの作り方（厳守）】
- 各シーンの start/end は、上の[開始秒〜終了秒]の値とそのまま一致させる（途中の秒で切らない）
- 連続して話がつながる場面は1シーンにまとめてよい。"○秒の間"がある所はシーンを分ける
- 並び順：シーンは時系列順（startの昇順）。前後に入れ替えない。同じ映像の重複は入れない
- start/end は 0以上{duration:.1f}以下
- 最後は自然に終わる場面で締める（文の途中で終わらない）
- bgm_mood は動画全体の感情に合うムードを1つ"""

def _structure_json(prompt, retries=2):
    """構成生成の1コール。APIエラーは握りつぶさず伝播させる（黙って劣化版を出さないため）。
    出力JSONが壊れている場合だけリトライし、最終的に有効な scenes 付きJSONを返す（無ければ None）。"""
    for _ in range(retries + 1):
        # API呼び出しの失敗（残高不足・認証・通信など）はそのまま例外として上へ伝える
        msg = anthropic_client.messages.create(
            model=STRUCTURE_MODEL, max_tokens=2500,
            messages=[{"role": "user", "content": prompt}])
        txt = msg.content[0].text.replace("```json", "").replace("```", "").strip()
        try:
            if json.loads(txt).get("scenes"):
                return txt
        except Exception:
            continue   # 出力が壊れている → リトライ
    return None

def analyze_content(annotated, duration):
    """編集の前に素材を"理解"する。種類・見どころ・締め・人間味・削る所を把握し、
    編集の前提（土台）にする。素材ごとのばらつきを抑え、毎回筋の通った判断をさせる狙い。"""
    prompt = (f"あなたはMagiClipの動画編集ディレクターです。次の{duration:.1f}秒の動画の文字起こし"
              f"（各行=1セグメント、[開始秒〜終了秒]と必要なら（注記）付き）を分析してください。\n\n{annotated}\n\n"
              "MagiClipは『話している内容は原則すべて残し、意味の通らない所だけ落とす』方針です。その前提で、"
              "次を簡潔に日本語の箇条書きで出力してください（JSON不要）:\n"
              "1. 動画の種類・テーマ（例：街歩き紹介／商品レビュー／トーク／ハウツー など）\n"
              "2. 全体で何を伝えている動画か（1〜2行）\n"
              "3. 落とすべき所（次だけを秒数で挙げる。無ければ「なし」）：\n"
              "   ・日本語として意味が通らない箇所（音声認識の崩れ・言い間違いで文にならない）\n"
              "   ・長い無音・言葉のない間\n"
              "   ・内容がまったく無い、フィラーだけの箇所\n"
              "   ※少しでも意味の通る発話・具体的な情報は、地味でもここに挙げない（残す）\n"
              "4. 終わりに自然な場面（最後の締めに適した一言。秒数）")
    try:
        msg = anthropic_client.messages.create(
            model=STRUCTURE_MODEL, max_tokens=1200,
            messages=[{"role": "user", "content": prompt}])
        return msg.content[0].text.strip()
    except Exception:
        return ""

GAP_SCENE_SPLIT = 0.8   # 残す発話どうしがこれ以上空いていたらシーンを分ける＝間の無音を除く


def _scenes_from_kept(cleaned_segments, drop):
    """『落とす』以外のセグメントを残してシーンを組み立てる（原則残す）。
    連続して残る発話は1シーンにまとめ、落とした所・長い無音（GAP_SCENE_SPLIT以上）で分ける
    ＝落とした箇所と無音だけが結果的にカットされる。"""
    scenes, last_kept = [], None
    for i, s in enumerate(cleaned_segments):
        if i in drop:
            continue
        if scenes and last_kept == i - 1 and (s["start"] - scenes[-1]["end"]) < GAP_SCENE_SPLIT:
            scenes[-1]["end"] = s["end"]                       # 直前の残す発話と地続き → 同じシーン
        else:
            scenes.append({"start": s["start"], "end": s["end"], "transition": "cut", "reason": ""})
        last_kept = i
    return scenes


def refine_transcript(cleaned_segments):
    """編集判断の前に、文字起こしを『意味の通る日本語』へ整える（文脈理解つき・テキスト起点編集の要）。
    各行を1:1で、明らかな音声認識の誤りだけ文脈から直す（統合/分割/並べ替え/要約はしない）。
    ここで整えたテキストで「残す/落とす」を判断する。返り値=整えた本文リスト（長さ・順序は入力と同一）。
    タイミングには使わない（＝同期に影響しない）。失敗時は原文をそのまま返す。"""
    if not cleaned_segments:
        return []
    numbered = "".join(f"[{i}] {s['text'].strip()}\n" for i, s in enumerate(cleaned_segments))
    prompt = f"""以下は動画の音声を自動認識した日本語の文字起こしです（各行[i]）。
話の流れ（文脈）を踏まえ、各行を「意味の通る自然な日本語」に整えてください。
目的は、この後の編集で「どこが伝えたい内容で、どこが意味を成さないか」を正しく判断できるようにすることです。

【やること】
- 明らかな音声認識の誤変換・言い間違いを、文脈から正しい語に直す（例：文脈上「5切るから」→「5千円切るから」）
- 固有名詞・地名・店名も、前後から明らかなら整える（不明なら変えない）

【禁止】
- 行の統合・分割・順序変更・要約・言い換えはしない（各行は1:1で対応）
- どうしても意味を復元できない行は、無理に作らずそのまま残す

[i]の個数・順序は入力と完全一致で、次の形式だけで返してください（他の文章は不要）：
[0] 整えた本文
[1] 整えた本文
...

文字起こし：
{numbered}"""
    try:
        msg = anthropic_client.messages.create(
            model=STRUCTURE_MODEL, max_tokens=3000,
            messages=[{"role": "user", "content": prompt}])
        out = msg.content[0].text.strip()
    except Exception:
        return [s["text"] for s in cleaned_segments]
    parts = re.split(r'\[(\d+)\]', out)
    parsed = {}
    for k in range(1, len(parts) - 1, 2):
        parsed[int(parts[k])] = ' '.join(re.sub(r'\[\d+\]', '', parts[k + 1]).split()).strip()
    return [parsed.get(i, "").strip() or s["text"] for i, s in enumerate(cleaned_segments)]


def generate_structure(cleaned_segments, duration, prev_choices=None, note=""):
    """『原則すべて残し、落とすものだけ選ぶ』方式で編集する（keep-by-default・テキスト起点）。
    まず文字起こしを整えて文脈を理解し、その"意味の通るテキスト"で「残す/落とす」を判断する。
    "良い所を選び抜く"のではなく、意味の通らない所・無音・空フィラーだけを落とす。
    ※整えたテキストは編集判断にだけ使い、start/endは元の値のまま（字幕の同期に影響しない）。"""
    if not cleaned_segments:
        return '{"scenes": []}'
    rules = _shared_rules(duration)

    # 0) テキストを整えて文脈を理解する（テキスト起点編集の起点）。start/endは元のまま
    refined_texts = refine_transcript(cleaned_segments)
    refined = [{"start": s["start"], "end": s["end"], "text": refined_texts[i],
                "avg_logprob": s.get("avg_logprob", 0.0)} for i, s in enumerate(cleaned_segments)]

    # 1) 整えたテキストで素材を理解する（編集の前提）
    annotated = _annotate_segments(refined)
    analysis = analyze_content(annotated, duration)
    analysis_block = f"# この素材の分析（編集の前提として必ず踏まえる）\n{analysis}\n\n" if analysis else ""

    note_block = f"\n【もう一度生成】{note}（ただし意味の通る発話は今回も残す）\n" if note else ""
    numbered = "\n".join(
        f"[{i}] {s['start']:.1f}-{s['end']:.1f}秒 {s['text']}" for i, s in enumerate(refined))

    # 2) 「落とすセグメント」だけを選ばせる（挙げなければ残る＝原則残す）
    prompt = (f"あなたはMagiClipのショート動画編集者です。下の【MagiClipの編集方針】に従ってください。\n\n"
              f"{rules}\n\n{analysis_block}"
              f"次は{duration:.1f}秒の動画の文字起こしです（各行=1セグメント）。\n{numbered}\n{note_block}\n"
              f"この中から『落とすセグメント』の番号だけを選んでください。挙げなかったものは全部残します（原則残す）。\n"
              f"落としてよいのは【日本語として意味が通らない（音声認識の崩れ・言い間違いで文にならない）／"
              f"長い無音／内容がまったく無いフィラーだけ】に明確に当てはまるものだけ。\n"
              f"重要度・面白さ・価値では判断しない（判断基準は『日本語の文として成立しているか』だけ）。"
              f"前後で1つの話は片方だけ落とさない。迷ったら残す（挙げない）。ほとんどは落とすもの無しが正常。\n"
              f'次のJSONのみで答えてください：{{"drop": [番号, ...], "bgm_mood": "動画に合うムードを1つ"}}')

    drop, bgm_mood = set(), ""
    for _ in range(3):
        # API失敗（残高・認証・通信）は握りつぶさず伝播。JSON崩れだけリトライ
        msg = anthropic_client.messages.create(
            model=STRUCTURE_MODEL, max_tokens=800,
            messages=[{"role": "user", "content": prompt}])
        txt = msg.content[0].text.replace("```json", "").replace("```", "").strip()
        try:
            data = json.loads(txt)
            drop = {int(x) for x in data.get("drop", []) if 0 <= int(x) < len(cleaned_segments)}
            bgm_mood = data.get("bgm_mood", "") or ""
            break
        except Exception:
            continue

    scenes = _scenes_from_kept(refined, drop)          # start/endは元の値（refined＝同じ時刻）
    if not scenes:                                     # 万一全部落とした→原則残す（安全側）
        scenes = _scenes_from_kept(refined, set())

    # 3) 自己チェック：日本語として意味が通らない場面だけが残っていれば除去（意味の通る発話は触らない）
    if analysis:
        scenes = refine_scenes(scenes, analysis, refined, duration)
    return json.dumps({"scenes": scenes, "bgm_mood": bgm_mood}, ensure_ascii=False)


def _scene_text(sc, cleaned):
    """シーン範囲に含まれるセグメント本文をつなげる（自己チェックの判断材料）。"""
    return " ".join(s["text"] for s in cleaned
                    if s["start"] >= sc["start"] - 0.3 and s["start"] < sc["end"])


def refine_scenes(scenes, analysis, cleaned, duration):
    """選定後の自己チェック（③）。日本語として意味が通らないシーンだけを除去する。
    - 対象：音声認識の崩れ・言い間違いで文にならない、情報として機能しない場面のみ
    - 話している内容は原則残す方針。少しでも意味が通る発話・情報・人間味・締めは絶対に残す
    - 減らしすぎる結果になったら適用しない（安全網）。時系列は保つ。"""
    if not analysis or len(scenes) <= 3:
        return scenes
    n = len(scenes)
    listing = "\n".join(
        f"[{i}] {sc['start']:.1f}-{sc['end']:.1f}秒: {_scene_text(sc, cleaned)}"
        for i, sc in enumerate(scenes))
    prompt = f"""あなたはMagiClipの編集品質チェック担当です。選ばれたシーンの中から、
「日本語として意味が通らず、視聴者に何も伝わらないシーン」だけをピンポイントで見つけてください。
MagiClipは"話している内容は原則すべて残す"方針です。これは最終確認であり、大掃除ではありません。迷ったら必ず残す。

# 選ばれたシーン（時系列順・[番号]）
{listing}

【除去してよいのは、次に“はっきり”当てはまるシーンだけ】
- 音声認識の誤変換・言い間違いで文が成立せず、何を言っているか分からない場面
  （例：「またリモコンやるのは奥川家浜吉」のように語がつながらず意味を成さない）

【除去してはいけない（重要・これらは地味でも必ず残す）】
- 意味の通る発話は全て残す。お店・地名・施設の紹介、感想、注意、豆知識などの具体的な情報
- 「一旦こんな感じ」「以上です」等の区切りの言葉も、話者の言葉なので残す（丸ごと消さない）
- 冒頭（導入）と最後のシーン（締め）は必ず残す
- 少しでも意味が取れるもの、判断に迷うものは残す

ほとんどの動画では除去ゼロが正常です。明確に意味不明なものが無ければ空で答えてください。
除去すべきシーンの番号だけをJSONで（他の文章は不要）：
{{"remove": [番号, ...]}}"""
    try:
        msg = anthropic_client.messages.create(
            model=STRUCTURE_MODEL, max_tokens=200,
            messages=[{"role": "user", "content": prompt}])
        txt = msg.content[0].text.replace("```json", "").replace("```", "").strip()
        remove = {int(x) for x in json.loads(txt).get("remove", [])}
    except Exception:
        return scenes                        # チェック失敗時は選定をそのまま使う（安全側）
    remove.discard(0)                        # 冒頭（導入）は現状維持＝守る
    remove.discard(n - 1)                    # 最後のシーン＝締めは必ず守る
    # 安全網：除去は最大でも全体の1/4まで（やり過ぎ防止）。超えたら信用せず適用しない
    if len(remove) > max(1, n // 4):
        return scenes
    kept = [sc for i, sc in enumerate(scenes) if i not in remove]
    if len(kept) < 3:
        return scenes
    return kept

def remove_overlapping_scenes(scenes):
    """時系列順に並べ、重複（同じ映像の二重表示）を除く。
    物語性を重視し、場面の順序は時系列を保つ（フックの前出し＝並べ替えはしない）。"""
    scenes = sorted(scenes, key=lambda x: x["start"])
    cleaned = []
    last_end = -1
    for scene in scenes:
        if scene["start"] >= last_end:
            cleaned.append(scene)
            last_end = scene["end"]
        elif scene["end"] > last_end:
            scene["start"] = last_end
            cleaned.append(scene)
            last_end = scene["end"]
    return cleaned

FADE_DUR = 0.15

def build_fade_filters(scene_duration):
    fade_out_st = max(0.0, scene_duration - FADE_DUR)
    vf = f"fade=t=in:st=0:d={FADE_DUR},fade=t=out:st={fade_out_st:.3f}:d={FADE_DUR}"
    af = f"afade=t=in:st=0:d={FADE_DUR},afade=t=out:st={fade_out_st:.3f}:d={FADE_DUR}"
    return vf, af

def probe_duration(path):
    """生成済みクリップの実際の長さを取得（字幕タイミングのマッピング用）"""
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream=duration", "-of", "csv=p=0", path],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return None

def fallback_scenes(cleaned_segments, duration):
    """Claudeの構成が使えないときの安全策：頭から自然に約35秒ぶん採用する"""
    scenes, total = [], 0.0
    for seg in cleaned_segments:
        scenes.append({"start": seg["start"], "end": seg["end"], "transition": "cut"})
        total += seg["end"] - seg["start"]
        if total >= 35:
            break
    if not scenes:
        scenes = [{"start": 0.0, "end": min(30.0, duration), "transition": "cut"}]
    return scenes

def parse_scenes(structure_json, cleaned_segments, duration):
    """Claude出力からシーンとBGMムードを取り出す。不正・空ならフォールバックする"""
    bgm_mood = ""
    raw = []
    try:
        data = json.loads(structure_json)
        bgm_mood = data.get("bgm_mood", "") or ""
        raw = data.get("scenes", []) or []
    except Exception:
        raw = []

    valid = []
    for sc in raw:
        try:
            s = max(0.0, min(float(sc["start"]), duration))
            e = max(0.0, min(float(sc["end"]), duration))
        except (KeyError, ValueError, TypeError):
            continue
        if e - s >= 0.5:
            trans = sc.get("transition", "cut")
            valid.append({"start": s, "end": e, "transition": "fade" if trans == "fade" else "cut"})

    valid = remove_overlapping_scenes(valid)
    if not valid:
        valid = fallback_scenes(cleaned_segments, duration)
    return valid, bgm_mood

def edit_video(video_path, scenes, output_path):
    temp_files = []
    timeline = 0.0
    for i, scene in enumerate(scenes):
        scene_duration = scene["end"] - scene["start"]
        temp_file = f"{OUTPUT_FOLDER}/temp_scene_{i}.mp4"
        # -ss / -to は -i の前（入力シーク）に置く。フェードを切り出し区間の先頭=0秒基準で効かせるため
        cmd = ["ffmpeg", "-ss", str(scene["start"]), "-to", str(scene["end"]), "-i", video_path]
        if scene.get("transition", "cut") == "fade":
            vf, af = build_fade_filters(scene_duration)
            cmd += ["-vf", vf, "-af", af]
        cmd += ["-c:v", "libx264", "-c:a", "aac", temp_file, "-y"]
        subprocess.run(cmd, capture_output=True)
        temp_files.append(temp_file)
        # 編集後タイムライン上の位置を実測長で記録（字幕マッピング用）
        actual = probe_duration(temp_file) or scene_duration
        scene["out_start"] = timeline
        scene["out_end"] = timeline + actual
        timeline += actual
    concat_list = f"{OUTPUT_FOLDER}/concat_list.txt"
    with open(concat_list, "w") as f:
        for temp_file in temp_files:
            f.write(f"file '{os.path.abspath(temp_file)}'\n")
    subprocess.run(["ffmpeg", "-f", "concat", "-safe", "0", "-i", concat_list, "-vf", "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2", "-c:v", "libx264", "-c:a", "aac", output_path, "-y"], capture_output=True)
    for temp_file in temp_files:
        os.remove(temp_file)
    os.remove(concat_list)
    return scenes

# ===== BGM（Pixabay等のCC0音源 / クレジット表記不要 / bgm/ に手動配置）=====
# Pixabay( https://pixabay.com/music/ )などからCC0のmp3をダウンロードし、
# 下記カテゴリ名で bgm/ フォルダに置いてください（例：bgm/bright.mp3）。
#   bright / emotional / tense / stylish / comical  ＋ 任意の default.mp3（フォールバック）
BGM_FOLDER = "bgm"
os.makedirs(BGM_FOLDER, exist_ok=True)
MOOD_KEYWORDS = {
    "bright":    ["明る", "楽し", "ポップ", "元気", "ハッピー", "happy", "bright", "fun", "upbeat"],
    "emotional": ["感動", "エモ", "壮大", "感情", "泣", "epic", "emotional", "inspir", "moving"],
    "tense":     ["緊張", "シリアス", "ドラマ", "迫力", "tense", "serious", "drama", "suspense"],
    "stylish":   ["おしゃれ", "オシャレ", "chill", "リラックス", "落ち着", "穏やか", "lofi", "lo-fi", "stylish", "calm", "relax"],
    "comical":   ["コミカル", "ユーモア", "面白", "ふざけ", "コメディ", "funny", "comic", "comical", "quirky"],
}

def bgm_category(mood):
    m = (mood or "").lower()
    for cat, keywords in MOOD_KEYWORDS.items():
        if any(kw.lower() in m for kw in keywords):
            return cat
    return "bright"

def get_bgm(mood):
    """ムードに合うCC0 BGMを bgm/ フォルダから探す（無ければNone）"""
    category = bgm_category(mood)
    for name in (category, "default"):
        path = os.path.join(BGM_FOLDER, f"{name}.mp3")
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return path, name
    return None, None


# ===== 字幕フォント：物語・映像の雰囲気に合わせて自動で選ぶ =====
# フォントが変わると映像の印象は大きく変わる。AIが既に出している mood をそのまま使うので
# 追加のAI呼び出し・コストはゼロ。フォントが無い環境（本番Linux等）では既定フォントに落ちる。
# (パス, ttcのindex, ユーザー向けの呼び名, 文字サイズ倍率, 黒フチの太さ, 可変フォントのウェイト名)
# ※フチは視認性の生命線。線の細い書体（明朝・丸ゴ）はフチを太くして背景に負けないようにする。
# ※すべて fonts/ に同梱した SIL Open Font License の書体（商用利用・動画への埋め込み可）。
#   OSにインストールされた書体に依存しないので、macでも本番Linuxでも同じ見た目になる。
FONT_MOODS = {
    "bright":    (f"{FONT_DIR}/ZenMaruGothic-Bold.ttf",     0, "丸ゴシック（親しみ・明るい）",  1.00, 4, None),
    "comical":   (f"{FONT_DIR}/DelaGothicOne-Regular.ttf",  0, "ポップ体（コミカル・元気）",    0.92, 3, None),
    "emotional": (f"{FONT_DIR}/ShipporiMinchoB1-Bold.ttf",  0, "明朝体（情感・余韻）",          1.04, 5, None),
    "tense":     (f"{FONT_DIR}/NotoSansJP[wght].ttf",       0, "太ゴシック（迫力・緊張感）",    1.00, 3, "Black"),
    "stylish":   (f"{FONT_DIR}/ZenKakuGothicNew-Bold.ttf",  0, "モダンゴシック（おしゃれ）",    1.00, 3, None),
}
_NOTO = f"{FONT_DIR}/NotoSansJP[wght].ttf"
DEFAULT_FONT = ((_NOTO, 0, "標準ゴシック（読みやすい）", 1.00, 3, "Bold") if os.path.exists(_NOTO)
                else (SUBTITLE_FONT, 0, "標準ゴシック（読みやすい）", 1.00, 3, None))


def _matched_mood(mood):
    """moodがキーワードに実際に当たったカテゴリを返す（当たらなければNone）。
    BGMは既定brightで良いが、フォントは当たらなければ『無難な標準ゴシック』に落としたいので分ける。"""
    m = (mood or "").lower()
    for cat, keywords in MOOD_KEYWORDS.items():
        if any(kw.lower() in m for kw in keywords):
            return cat
    return None


def pick_subtitle_font(mood):
    """動画のムードに合う字幕フォントを選ぶ → (path, index, label, scale, stroke)。
    ムード不明・フォント未導入なら既定の標準ゴシック（迷ったら無難に、が最優先）。"""
    cat = _matched_mood(mood)
    ent = FONT_MOODS.get(cat) if cat else None
    if ent and ent[0] and os.path.exists(ent[0]):
        return ent
    return DEFAULT_FONT


def font_options():
    """プロ編集で選べる書体の一覧 → [(key, label)]。この環境に無い書体は出さない。"""
    opts = [("auto", "AIにおまかせ")]
    for key, ent in FONT_MOODS.items():
        if ent[0] and os.path.exists(ent[0]):
            opts.append((key, ent[2]))
    opts.append(("default", DEFAULT_FONT[2]))
    return opts


def font_by_key(key, mood=""):
    """キー指定の書体を返す。'auto'/未知/未導入なら雰囲気からの自動選択にフォールバック。"""
    if key and key != "auto":
        if key == "default":
            return DEFAULT_FONT
        ent = FONT_MOODS.get(key)
        if ent and ent[0] and os.path.exists(ent[0]):
            return ent
    return pick_subtitle_font(mood)

def mix_bgm(video_path, bgm_path, output_path, volume=0.15):
    result = subprocess.run([
        "ffmpeg", "-i", video_path, "-stream_loop", "-1", "-i", bgm_path,
        "-filter_complex", f"[1:a]volume={volume}[bg];[0:a][bg]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[a]",
        "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-shortest",
        output_path, "-y"
    ], capture_output=True)
    return result.returncode == 0

CORRECT_MIN_RATIO = 0.35   # 校正結果が元とこれ未満しか一致しなければ「別物に化けた」とみなし元を採用

def correct_segments(segments, reference=""):
    """字幕テキストを"控えめに"整える。最優先は『音声と字幕が一致していること』。
    各行を1:1で軽く整えるだけ（言い換え・要約・統合・並べ替えは禁止）。誤字脱字とフィラーのみ補正。
    reference は崩れた語の確認だけに使う（本文に持ち込まない）。
    入力・出力ともに [{start, end, text}] のリスト（タイミングは変えない）。"""
    if not segments:
        return []
    numbered = ""
    for i, seg in enumerate(segments):
        numbered += f"[{i}] {seg['text'].strip()}\n"
    ref_block = (f"\n【参考：動画全体の文字起こし】（崩れた語を確認するためだけに使う。ここから本文に語を持ち込まない）\n{reference}\n"
                 if reference else "")
    try:
        message = anthropic_client.messages.create(
            model=SUBTITLE_MODEL,
            max_tokens=2000,
            messages=[{"role": "user", "content": f"""
以下は動画の音声を自動認識した日本語字幕です。各行 [i] の番号・行数・順序を必ず保ち、テキストを"控えめに"整えてください。
最優先は「音声と字幕が一致していること」。話した言葉のまま残すのが基本です。
{ref_block}
【やること（控えめに）】
- 明らかな誤字・変換ミスだけ直す（例：「以外と」→「意外と」「気ずく」→「気づく」「下さい」→「ください」）
- 行頭の言い淀み・フィラーだけ削る（えーと/あの/その/えー/まあ/なんか）。本文の語は残す
- 同じ行内の直後の重複だけ1つにする（例：「ローソン ローソンが」→「ローソンが」）。行をまたぐ統合はしない
- 句読点（。や、）は付けない。区切りは半角スペース

【禁止（字幕ズレの原因になる）】
- 要約・言い換え・短縮で別の表現にしない（実際に話した言葉のまま。長くても短くしない）
- 行をまとめる・分ける・順番を変える・別の行の内容を持ち込まない
- 入力に無い語を足さない（上の参考は崩れた語の"確認"だけ。本文に持ち込まない）
- 固有名詞・数字は確信が無ければ変えない
- [i] の個数・順序は入力と完全一致。各 [i] は同じ [i] の入力だけを直したものにする

テキスト：
{numbered}
"""}]
        )
        out = message.content[0].text.strip()
    except Exception:
        return [dict(s) for s in segments]  # 校正失敗時は原文のまま（ズレを出さない）

    # [i] で分割して index→テキスト の対応表を作る（マーカー欠落でも後続を巻き込まない）
    parts = re.split(r'\[(\d+)\]', out)
    parsed = {}
    for k in range(1, len(parts) - 1, 2):
        text = re.sub(r'\[\d+\]', '', parts[k + 1])
        parsed[int(parts[k])] = ' '.join(text.split()).strip()

    corrected = []
    for i, seg in enumerate(segments):
        orig = seg["text"].strip()
        new = parsed.get(i, "").strip()
        # ★安全網1：元と大きく食い違ったら（別の行に化けた・幻字幕）元を採用＝ズレ防止
        if not new or difflib.SequenceMatcher(None, orig, new).ratio() < CORRECT_MIN_RATIO:
            new = orig
        # ★安全網2：文字数が増えたら（＝行の結合・語の追加）元を採用。校正は短くなるはずで、増えるのは異常
        elif len(new.replace(" ", "")) > len(orig.replace(" ", "")) + 4:
            new = orig
        corrected.append({"start": seg["start"], "end": seg["end"], "text": new,
                          "orig_start": seg.get("orig_start"), "orig_end": seg.get("orig_end")})
    return corrected

PAUSE_SPLIT = 0.4     # 単語間がこれ以上空いたら「話の区切り」→字幕を分ける（音声と同期させる肝）
CUE_MAX_DUR = 8.0     # 1キューの最大表示秒数（暴走防止の安全網）
CUE_SOFT_CHARS = 13   # この長さを超えたら、次の"自然な区切り"（助詞・文末）で切る → 短く読みやすく
CUE_HARD_CHARS = 22   # 強制上限。縦型ショートは2行×約11字までが読みやすい（Web調査）ため低めに
CUE_MIN_SHOW = 0.7    # 1キューの最低表示秒数（一瞬で消えないように）
# キューを切ってよいのは"文末系"だけにする（句読点＋終助詞ね/よ/わ）。
# が/を/に/は/も 等の格助詞で切ると述語が次行に孤立する（「ファミマが」｜「あります」）ため使わない。
# と/の/て/で 等は語の途中(という・ところ・やって・ので…)に出るため使わない。
_NATURAL_BREAK = set("、。！？ねよわ")

def build_word_cues(words, scenes):
    """原音声の単語タイムスタンプを編集後タイムラインへマッピングして字幕を作る。
    字幕は「実際に喋りが止まった所（間）」と「文末」でだけ区切る → 音声と同期し、改行も自然になる。
    各キューは所属シーンの編集後区間[out_start,out_end]にクランプし、ズレがシーンを越えて広がらないようにする。"""
    cues = []
    for scene in scenes:
        s, e = scene["start"], scene["end"]
        out_s, out_e = scene["out_start"], scene["out_end"]
        cur, c_start, c_end, prev_end = "", None, None, None

        def emit():
            nonlocal cur, c_start, c_end
            text = cur.strip("、。 　")
            if text and c_start is not None:
                cs = max(out_s, min(c_start - s + out_s, out_e))   # シーン区間にクランプ（負値・はみ出し防止）
                ce = max(cs, min(c_end - s + out_s, out_e))
                # orig_start/orig_end = 元動画上の時刻（テキストベース編集でカット位置を割り出すのに使う）
                cues.append({"start": cs, "end": ce, "text": text,
                             "orig_start": c_start, "orig_end": c_end})
            cur, c_start, c_end = "", None, None

        for w in words:
            ws = getattr(w, "start", None)
            if ws is None or ws < s - 0.05 or ws >= e:
                continue
            token = w.word
            if token.strip("、。．！？!?・ 　") in FILLER_TOKENS:   # フィラー語は表示しない
                continue
            we = getattr(w, "end", ws)
            # 直前の単語との間が空いていたら「話の区切り」→ ここまでを確定（音声に同期して切り替える）
            if c_start is not None and prev_end is not None and (ws - prev_end) >= PAUSE_SPLIT:
                emit()
            if c_start is None:
                c_start = ws
            cur += token
            c_end = max(we, c_start)
            prev_end = we
            last = cur.rstrip("　 ")[-1:]
            # 確定タイミング：文末／長すぎ(安全網)／ソフト上限を超えて"自然な区切り(助詞・句読点)"に来た
            # ※区切りは必ず助詞・文末＝語の途中では切らない（読みやすさ）。文字数だけでは切らない（同期維持）
            if (token.rstrip().endswith(("。", "．", "！", "？", "!", "?"))
                    or (c_end - c_start) >= CUE_MAX_DUR
                    or len(cur) >= CUE_HARD_CHARS
                    or (len(cur) >= CUE_SOFT_CHARS and last in _NATURAL_BREAK)):
                emit()
        emit()

    cues.sort(key=lambda x: x["start"])
    result = []
    for c in cues:
        if c["end"] <= c["start"]:
            c["end"] = c["start"] + 0.4
        # 連続する言い直し・重複を間引く（読みやすさ）：直前と同じ/直前に含まれるなら出さない。
        # 直前が今回に含まれる短い言い直し（例「六方会館」→「六方会館ですね」）は今回で置き換える。
        if result:
            prev = result[-1]["text"]
            if c["text"] == prev or c["text"] in prev:
                continue
            if prev in c["text"] and len(prev) <= 8:
                result[-1] = c
                continue
        if result and c["start"] < result[-1]["end"]:
            result[-1]["end"] = c["start"]   # 重なり解消（同時に2つ表示しない）
        result.append(c)
    # 最低表示時間を確保（次の字幕の開始は侵さない範囲で延長）→ 一瞬で消えるのを防ぐ
    for i, c in enumerate(result):
        limit = result[i + 1]["start"] if i + 1 < len(result) else c["end"] + CUE_MIN_SHOW
        if c["end"] - c["start"] < CUE_MIN_SHOW:
            c["end"] = min(c["start"] + CUE_MIN_SHOW, limit)
    return result

def dedup_consecutive_subs(subs):
    """連続する重複字幕を間引く（話者の言い直しを校正が同文言にそろえても重複させない）。
    直前と同じ／直前に含まれる字幕は出さない＝最初の1つだけ残す（時刻は最初の発話に合う）。"""
    out = []
    for s in subs:
        if out and (s["text"] == out[-1]["text"] or s["text"] in out[-1]["text"]):
            continue
        out.append(s)
    return out

def sanitize_caption(text):
    """字幕に紛れ込む校正マーカーや非発話の記号・注記を確実に除去する（多層防御）。
    AI校正が稀に [12] 等のマーカーや（笑）(BGM)等の注記を残しても、SRTには出さない。"""
    if not text:
        return ""
    text = re.sub(r'\[\s*\d+\s*\]', ' ', text)          # 校正の番号マーカー [12]
    text = re.sub(r'【[^】]{0,12}】', ' ', text)          # 隅付き括弧の注記【BGM】ごと除去
    text = re.sub(r'[（(][^（）()]{0,10}[）)]', ' ', text)  # 効果音・注記の括弧（笑）(BGM)
    text = re.sub(r'[\[\]【】]', ' ', text)              # はみ出した括弧の残り
    text = re.sub(r'[♪♫♬◆◇■□*※→←]', ' ', text)        # 音符・装飾記号
    text = re.sub(r'^[\s、。,.\-‐－—–…・:：;；]+', '', text)  # 行頭の接続記号・約物
    text = re.sub(r'^(と|で)[\s、。,]+', '', text)       # 行頭に浮く接続詞「と/で」（直後が空白/読点＝単独の繋ぎ語）を除去
    return ' '.join(text.split())

def strip_punct(text):
    """字幕は句読点なしの方が見やすい。。、を除き、区切りは半角スペースにする。"""
    for p in ("、", "。", "，", "．", "､", "｡"):
        text = text.replace(p, " ")
    return " ".join(text.split())

def create_srt(segments, output_srt):
    def to_srt_time(seconds):
        seconds = max(0.0, seconds)   # 負の時刻を防ぐ（-1:59:59 のような不正表記を回避）
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"
    with open(output_srt, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments):
            f.write(f"{i+1}\n{to_srt_time(segment['start'])} --> {to_srt_time(segment['end'])}\n{segment['text']}\n\n")

# 改行位置の良し悪し（読みやすさ）の判定に使う文字
_BREAK_AFTER = "、。！？" + "はがをにへとでもやのからまでねよさ"   # この直後で改行すると自然（句読点・助詞）
_BAD_LINE_HEAD = "、。！？ー〜ゃゅょっぁぃぅぇぉ・"                  # 行頭に来てほしくない文字
# 節（文節）の切れ目になりやすい語末。ここで切ると意味のまとまりで改行できる（ひらがな連続でも切ってよい）
_CLAUSE_END = ("けど", "けれど", "から", "ので", "のに", "たら", "なら", "って",
               "です", "ます", "ました", "でした", "ですね", "ますね", "だけど", "だから")
# 縦型ショート向けの1行の目安（Web調査：縦型は1行8〜10字・最大2行・文節で改行が読みやすい）
LINE_CHARS_MAX = 12

_TOKENIZER = None
_JIRITSU = {"名詞", "動詞", "形容詞", "副詞", "連体詞", "接続詞", "感動詞"}   # 自立語（文節の先頭になれる）

def _break_positions(text):
    """形態素解析で「文節の切れ目」の文字位置を返す（ここでしか改行しない）。
    付属語（助詞・助動詞・接尾・非自立語＝な/です/が/ところ/さん 等）は前の語にくっつけ、
    自立語（名詞・動詞など）の直前でだけ改行できるようにする＝『有名｜な』のような切れを防ぐ。
    janomeが使えない環境では None を返し、従来の文字ベース判定にフォールバックする。"""
    global _TOKENIZER
    try:
        if _TOKENIZER is None:
            from janome.tokenizer import Tokenizer
            _TOKENIZER = Tokenizer()
        positions, total = set(), 0
        for tok in _TOKENIZER.tokenize(text):
            parts = tok.part_of_speech.split(",")
            pos0 = parts[0]
            pos1 = parts[1] if len(parts) > 1 else ""
            # 自立語で、かつ付属的(非自立・接尾)でなければ「新しい文節の先頭」＝その手前で改行可
            if total > 0 and pos0 in _JIRITSU and pos1 not in ("非自立", "接尾"):
                positions.add(total)
            total += len(tok.surface)
        if total != len(text):          # 解析結果が元と長さ不一致なら信用しない
            return None
        # 空白（こちらが入れた区切り）は必ず改行候補に含める
        for idx, ch in enumerate(text):
            if ch == " ":
                positions.add(idx); positions.add(idx + 1)
        return {p for p in positions if 0 < p < len(text)}
    except Exception:
        return None

def wrap_text(text, font, max_width, draw):
    """字幕を読みやすく改行する。縦型ショート基準で、1行が長い(=12字超)場合も2行に分ける。
    改行は形態素解析で得た「単語の切れ目」だけで行い、語の途中では絶対に切らない。
    その中から「空白＞句読点＞文節末(です/けど等)＞助詞」を優先し、左右の長さを揃える。"""
    def width(s):
        b = draw.textbbox((0, 0), s, font=font)
        return b[2] - b[0]

    # 1行に収まり、かつ十分短ければそのまま1行（縦型は1行を短く保つほど読みやすい）
    if width(text) <= max_width and len(text) <= LINE_CHARS_MAX:
        return [text]

    n = len(text)
    target = n / 2
    breaks = _break_positions(text)                 # 単語境界（無ければ従来通り全位置）
    candidates = sorted(breaks) if breaks else range(1, n)
    best = None
    def _kata(c): return "゠" <= c <= "ヿ"
    def _kanji(c): return "一" <= c <= "鿿"
    def _hira(c): return "ぁ" <= c <= "ゟ"
    for i in candidates:
        if not (0 < i < n):
            continue
        if width(text[:i]) > max_width or width(text[i:]) > max_width:
            continue
        prev, nxt = text[i - 1], text[i]
        score = abs(i - target)            # 中央に近いほど良い（左右の文字数を揃える＝最優先）
        natural = True
        if prev == " " or nxt == " ":
            score -= 20                    # 空白（＝こちらが入れた区切り）は最優先で切る
        elif breaks is not None:
            # 文節モード：候補はどれも良い区切り。バランス優先で、区切り種別は軽い加点のみ
            if prev in "、。！？":
                score -= 3
            elif any(text[:i].endswith(e) for e in _CLAUSE_END):
                score -= 2                 # 文節・節の切れ目を少しだけ優遇
            elif prev in _BREAK_AFTER:
                score -= 1
        else:
            # フォールバック（文字ベース）：強めに自然な区切りへ誘導＋語中で割るのを防ぐ
            if prev in "、。！？":
                score -= 12
            elif any(text[:i].endswith(e) for e in _CLAUSE_END):
                score -= 11
            elif prev in _BREAK_AFTER:
                score -= 4
            else:
                natural = False
            if not natural:
                if _kata(prev) and _kata(nxt):
                    score += 15
                elif _kanji(prev) and _kanji(nxt):
                    score += 5
                elif _hira(prev) and _hira(nxt):
                    score += 10
        if nxt in _BAD_LINE_HEAD or prev in "ー":
            score += 8                     # 行頭に小書き・長音等が来る改行は避ける（見た目）
        la, lb = len(text[:i].strip()), len(text[i:].strip())
        if max(la, lb) > LINE_CHARS_MAX:
            score += (max(la, lb) - LINE_CHARS_MAX) * 2   # 1行が長すぎる改行を避ける（縦型は1行を短く）
        if lb <= 1 or la <= 1:
            score += 6                     # 片方が極端に短い改行は避ける（孤立防止）
        if best is None or score < best[0]:
            best = (score, i)
    if best:
        i = best[1]
        return [text[:i].strip(), text[i:].strip()]   # 区切りが空白なら前後の空白を除く

    # 2行で収まらない長文は機械的に折り返す（通常は起きない）
    lines, cur = [], ""
    for ch in text:
        if width(cur + ch) <= max_width:
            cur += ch
        else:
            lines.append(cur)
            cur = ch
    if cur:
        lines.append(cur)
    return lines

def render_subtitle_png(text, w, h, path, font_choice=None):
    """字幕1枚を透過PNGとして描画（下部中央・白文字＋黒フチ）。
    font_choice=(path,index,label,scale) で雰囲気に合わせた書体を使う（Noneなら既定）。"""
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    fpath, fidx, _label, scale, stroke, variation = font_choice or DEFAULT_FONT
    font_size = int(52 * scale)   # ショート向けに大きめ（短いキューと合わせて読みやすく）
    try:
        font = load_font(fpath, font_size, fidx, variation) if fpath else ImageFont.load_default()
    except Exception:
        try:   # 選んだ書体が使えない環境では既定フォントに落ちる（字幕を失わない）
            font = load_font(SUBTITLE_FONT, font_size) if SUBTITLE_FONT else ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()
    lines = wrap_text(text, font, int(w * 0.90), draw)
    line_height = font_size + 12
    start_y = h - 140 - line_height * len(lines)

    # 明るい背景でも読めるように、文字の下に“やわらかい影”を敷く。
    # 黒帯を置くと画が隠れて重くなるため、テキスト形状のぼかし影で背景の明度に依存させない。
    shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(shadow)
    y = start_y
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        x = (w - (bbox[2] - bbox[0])) // 2
        try:
            sdraw.text((x, y), line, font=font, fill=(0, 0, 0, 210),
                       stroke_width=stroke + 4, stroke_fill=(0, 0, 0, 210))
        except TypeError:
            sdraw.text((x, y), line, font=font, fill=(0, 0, 0, 210))
        y += line_height
    img.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(7)))

    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        x = (w - (bbox[2] - bbox[0])) // 2
        try:
            draw.text((x, start_y), line, font=font, fill=(255, 255, 255, 255),
                      stroke_width=stroke, stroke_fill=(0, 0, 0, 255))
        except TypeError:
            for dx in (-2, 0, 2):
                for dy in (-2, 0, 2):
                    draw.text((x+dx, start_y+dy), line, font=font, fill=(0, 0, 0, 255))
            draw.text((x, start_y), line, font=font, fill=(255, 255, 255, 255))
        start_y += line_height
    img.save(path)

def _render_watermark_png(w, h, path, alpha=150):
    """透かし「✦ MagiClip」を右下に描いた全画面透過PNGを作る（白・半透明）。
    ✦は環境フォント非依存で描けるようPILで4方向スパークを多角形描画する。"""
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    size = max(26, w // 36)            # 動画幅に合わせた文字サイズ
    try:
        font = load_font(SUBTITLE_FONT, size) if SUBTITLE_FONT else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()
    text = "MagiClip"
    tb = draw.textbbox((0, 0), text, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    spark = size // 2                  # スパークの半径
    gap = size // 4
    margin = max(36, w // 24)
    block_h = max(th, spark * 2)
    x = w - margin - (spark * 2 + gap + tw)
    y = h - margin - block_h
    white = (255, 255, 255, alpha)
    # ✦ スパーク（4方向の星）を描画
    cx, cy = x + spark, y + block_h // 2
    inner = spark * 0.36
    pts = [(cx, cy - spark), (cx + inner, cy - inner), (cx + spark, cy), (cx + inner, cy + inner),
           (cx, cy + spark), (cx - inner, cy + inner), (cx - spark, cy), (cx - inner, cy - inner)]
    draw.polygon(pts, fill=white)
    # ワードマーク
    draw.text((x + spark * 2 + gap, cy - th // 2 - tb[1]), text, font=font, fill=white)
    img.save(path)


def add_watermark(input_path, output_path):
    """動画の右下に「✦ MagiClip」の透かし（白・半透明）を焼き込む。成功で True。"""
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", input_path],
                           capture_output=True, text=True)
    try:
        w, h = (int(x) for x in probe.stdout.strip().split("x"))
    except ValueError:
        w, h = 1080, 1920
    wm = os.path.splitext(output_path)[0] + "_wm.png"
    _render_watermark_png(w, h, wm)
    r = subprocess.run(["ffmpeg", "-y", "-i", input_path, "-i", wm,
                        "-filter_complex", "[0:v][1:v]overlay=0:0",
                        "-c:a", "copy", output_path], capture_output=True)
    try:
        os.remove(wm)
    except OSError:
        pass
    return r.returncode == 0


def burn_subtitles(input_path, subtitles, output_path, font_choice=None):
    """字幕を透過PNGとしてoverlayで1パス合成する（全フレーム展開せず高速・無劣化）。
    font_choice で動画の雰囲気に合った書体を使う。"""
    if not subtitles:
        shutil.copy(input_path, output_path)
        return
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", input_path],
                           capture_output=True, text=True)
    try:
        w, h = (int(x) for x in probe.stdout.strip().split("x"))
    except ValueError:
        w, h = 1080, 1920

    png_paths, inputs, filter_parts = [], ["-i", input_path], []
    last = "0:v"
    for i, sub in enumerate(subtitles):
        png = f"{OUTPUT_FOLDER}/_sub_{i}.png"
        render_subtitle_png(sub["text"], w, h, png, font_choice)
        png_paths.append(png)
        inputs += ["-i", png]
        cur = f"v{i+1}"
        filter_parts.append(
            f"[{last}][{i+1}:v]overlay=enable='between(t,{sub['start']:.3f},{sub['end']:.3f})'[{cur}]"
        )
        last = cur

    cmd = ["ffmpeg", *inputs, "-filter_complex", ";".join(filter_parts),
           "-map", f"[{last}]", "-map", "0:a?", "-c:v", "libx264", "-c:a", "copy",
           output_path, "-y"]
    subprocess.run(cmd, capture_output=True)
    for p in png_paths:
        os.remove(p)

def process_video_auto(job_id, video_path):
    """全自動：アップロード→文字起こし→AI構成→編集→字幕→BGM→完成（編集画面なし）。"""
    try:
        job = jobs[job_id]
        job.update(status="処理中", progress=10)
        audio_path = f"{OUTPUT_FOLDER}/{job_id}_audio.mp3"

        job["progress"] = 25
        duration = get_video_duration(video_path)
        extract_audio(video_path, audio_path)

        job["progress"] = 40
        transcript = transcribe_audio(audio_path)
        cleaned = clean_segments(transcript)

        # 再生成で使い回せるよう保持（文字起こしは1回だけ）
        job["transcript"] = transcript
        job["cleaned"] = cleaned
        job["video_path"] = video_path
        job["duration"] = duration

        _generate_and_build(job_id)

    except Exception as e:
        jobs[job_id] = {"status": "エラー", "error": str(e)}

def _generate_and_build(job_id):
    """AI構成→編集→字幕→BGM→完成。再生成時はキャッシュした文字起こしを再利用し、
    generate_structure を呼び直すことで毎回ちがう構成（別パターン）になる。"""
    job = jobs[job_id]
    cleaned = job["cleaned"]
    transcript = job["transcript"]
    video_path = job["video_path"]
    duration = job["duration"]
    output_path = f"{OUTPUT_FOLDER}/{job_id}_output.mp4"
    srt_path = f"{OUTPUT_FOLDER}/{job_id}_subtitles.srt"
    subtitled_path = f"{OUTPUT_FOLDER}/{job_id}_subtitled.mp4"
    final_path = f"{OUTPUT_FOLDER}/{job_id}_final.mp4"

    job.update(status="生成中", progress=55)
    prev_choices = job.get("prev_choices", [])   # 過去に作った構成（作り直し時に削り具合を変える）
    note = REGEN_NOTES[len(prev_choices) % len(REGEN_NOTES)]   # 再生成のたびに削り具合だけ変える
    structure = generate_structure(cleaned, duration, prev_choices=prev_choices, note=note)
    scenes, bgm_mood = parse_scenes(structure, cleaned, duration)
    # 今回採用した構成を記録（次の作り直しで重複を避ける）
    job.setdefault("prev_choices", []).append([(round(s["start"], 1), round(s["end"], 1)) for s in scenes])
    edit_video(video_path, scenes, output_path)

    # 字幕：原音声の認識結果（テキストが高品質）を編集後タイムラインへマッピング。
    # タイミングは実測上ほぼズレない（診断で編集後音声の実時刻とほぼ一致を確認）。
    job["progress"] = 80
    cues = build_word_cues(transcript.words, scenes)
    # 動画全体の文字起こしを参考に渡し、崩れた語（数字・一般語）を文脈で直す
    reference = " ".join(s["text"].strip() for s in cleaned)
    subtitles = correct_segments(cues, reference=reference)
    for s in subtitles:                       # 余分な文字を除去＋句読点なしに（仕上げ）
        s["text"] = strip_punct(sanitize_caption(s["text"]))
    # 空・単独かな1文字（シーン境界で割れた助詞などの断片）は字幕に出さない
    _frag = re.compile(r'^[぀-ゟ゠-ヿ]$')
    subtitles = [s for s in subtitles if s["text"] and not _frag.match(s["text"])]
    subtitles = dedup_consecutive_subs(subtitles)   # 校正後の連続重複も除去
    create_srt(subtitles, srt_path)

    job["progress"] = 88
    font_choice = pick_subtitle_font(bgm_mood)   # 雰囲気に合った書体を選ぶ
    job["font"] = font_choice[2]
    burn_subtitles(output_path, subtitles, subtitled_path, font_choice)

    job["progress"] = 95
    bgm_path, _ = get_bgm(bgm_mood)
    if bgm_path and mix_bgm(subtitled_path, bgm_path, final_path):
        os.remove(subtitled_path)
    else:
        os.replace(subtitled_path, final_path)

    # 透かし「✦ MagiClip」を右下に焼き込む（仕上げ）
    wm_final = f"{OUTPUT_FOLDER}/{job_id}_wm_final.mp4"
    if add_watermark(final_path, wm_final):
        os.replace(wm_final, final_path)

    job.update(status="完成", progress=100, output=final_path)

def regenerate_job(job_id):
    """同じ動画で別パターンを生成（文字起こしは再利用）。"""
    try:
        job = jobs.get(job_id)
        if not job or "cleaned" not in job:
            return
        _generate_and_build(job_id)
    except Exception as e:
        jobs[job_id] = {"status": "エラー", "error": str(e)}

HTML = """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MagiClip — 動画AI編集</title>
    <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'><path d='M12 1 L14.2 9.8 L23 12 L14.2 14.2 L12 23 L9.8 14.2 L1 12 L9.8 9.8 Z' fill='%237C3AED'/></svg>">
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, sans-serif; background: #0f0f0f; color: #fff; min-height: 100vh; display: flex; align-items: center; justify-content: center; }
        .container { width: 100%; max-width: 480px; padding: 40px 20px; }
        h1 { font-size: 28px; font-weight: 700; margin-bottom: 8px; }
        .logo { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
        .logo-spark { width: 30px; height: 30px; fill: #7C3AED; flex-shrink: 0; filter: drop-shadow(0 0 7px rgba(124,58,237,.45)); }
        .logo-text { font-size: 30px; font-weight: 800; letter-spacing: .5px;
                     background: linear-gradient(135deg, #7C3AED, #A78BFA);
                     -webkit-background-clip: text; background-clip: text; color: transparent; }
        p.sub { color: #888; margin-bottom: 40px; font-size: 15px; }
        .upload-area { border: 2px dashed #333; border-radius: 16px; padding: 60px 20px; text-align: center; cursor: pointer; transition: all 0.2s; }
        .upload-area:hover { border-color: #555; background: #1a1a1a; }
        .upload-area input { display: none; }
        .upload-icon { font-size: 48px; margin-bottom: 16px; }
        .upload-text { color: #888; font-size: 15px; }
        .upload-text span { color: #fff; font-weight: 600; }
        .btn { width: 100%; padding: 16px; background: #fff; color: #000; border: none; border-radius: 12px; font-size: 16px; font-weight: 600; cursor: pointer; margin-top: 16px; transition: opacity 0.2s; }
        .btn:hover { opacity: 0.9; }
        .btn:disabled { opacity: 0.4; cursor: not-allowed; }
        .progress-area { display: none; margin-top: 32px; }
        .progress-bar { background: #222; border-radius: 100px; height: 6px; margin: 16px 0; }
        .progress-fill { background: #fff; height: 100%; border-radius: 100px; transition: width 0.3s; }
        .status-text { color: #888; font-size: 14px; text-align: center; }
        .download-area { display: none; margin-top: 32px; text-align: center; }
        .preview-video { width: 100%; max-height: 60vh; border-radius: 12px; background: #000; }
        .download-btn { display: inline-block; padding: 14px 24px; background: #22c55e; color: #fff; border-radius: 12px; text-decoration: none; font-weight: 600; font-size: 15px; }
        .btn-secondary { padding: 14px 24px; background: #222; color: #fff; border: 1px solid #444; border-radius: 12px; font-weight: 600; font-size: 15px; cursor: pointer; }
        .btn-secondary:hover { background: #2c2c2c; }
        .file-name { margin-top: 16px; color: #aaa; font-size: 14px; }
        .error-area { display: none; margin-top: 32px; padding: 16px; background: #2a1a1a; border-radius: 12px; color: #f87171; font-size: 14px; }
        .edit-area { display: none; margin-top: 28px; }
        .edit-area h2 { font-size: 17px; font-weight: 700; margin-bottom: 6px; }
        .edit-area .hint { color: #888; font-size: 13px; margin-bottom: 16px; }
        .sub-list { max-height: 50vh; overflow-y: auto; border: 1px solid #222; border-radius: 12px; padding: 8px; }
        .sub-row { display: flex; align-items: center; gap: 10px; padding: 6px 4px; }
        .sub-time { color: #666; font-size: 12px; font-variant-numeric: tabular-nums; min-width: 42px; text-align: right; }
        .sub-input { flex: 1; background: #1a1a1a; border: 1px solid #2c2c2c; color: #fff; border-radius: 8px; padding: 10px 12px; font-size: 15px; font-family: inherit; }
        .sub-input:focus { outline: none; border-color: #555; background: #222; }
        .scene-area { display: none; margin-top: 28px; }
        .scene-area h2 { font-size: 17px; font-weight: 700; margin-bottom: 6px; }
        .scene-area .hint { color: #888; font-size: 13px; margin-bottom: 16px; }
        .scene-list { max-height: 50vh; overflow-y: auto; border: 1px solid #222; border-radius: 12px; padding: 4px; }
        .scene-row { display: flex; gap: 10px; padding: 10px 8px; border-radius: 8px; cursor: pointer; align-items: flex-start; }
        .scene-row:hover { background: #1a1a1a; }
        .scene-row.on { background: #15241a; }
        .scene-row input { margin-top: 3px; width: 18px; height: 18px; flex-shrink: 0; accent-color: #22c55e; }
        .scene-row .thumb { width: 68px; height: 120px; flex-shrink: 0; border-radius: 6px; object-fit: cover; background: #222; }
        .scene-row .meta { flex: 1; }
        .scene-row .t { color: #666; font-size: 11px; font-variant-numeric: tabular-nums; margin-bottom: 4px; }
        .scene-row .stxt { width: 100%; background: #1a1a1a; border: 1px solid #2c2c2c; color: #fff; border-radius: 8px; padding: 8px 10px; font-size: 14px; font-family: inherit; line-height: 1.4; resize: vertical; }
        .scene-row .stxt:focus { outline: none; border-color: #22c55e; background: #222; }
    </style>
</head>
<body>
<div class="container">
    <div class="logo">
        <svg class="logo-spark" viewBox="0 0 24 24" aria-hidden="true">
            <path d="M12 1 L14.2 9.8 L23 12 L14.2 14.2 L12 23 L9.8 14.2 L1 12 L9.8 9.8 Z"/>
        </svg>
        <span class="logo-text">MagiClip</span>
    </div>
    <p class="sub">動画をアップロードするだけで、字幕付きショート動画を自動生成します</p>
    <div class="upload-area" onclick="document.getElementById('fileInput').click()">
        <input type="file" id="fileInput" accept="video/*" onchange="handleFileSelect(event)">
        <div class="upload-icon">🎬</div>
        <div class="upload-text"><span>動画を選択</span>またはここにドロップ</div>
        <div class="file-name" id="fileName"></div>
    </div>
    <button class="btn" id="uploadBtn" onclick="uploadVideo()" disabled>自動編集をはじめる</button>
    <div class="progress-area" id="progressArea">
        <div class="status-text" id="statusText">処理中...</div>
        <div class="progress-bar">
            <div class="progress-fill" id="progressFill" style="width: 0%"></div>
        </div>
    </div>
    <div class="download-area" id="downloadArea">
        <p style="color:#22c55e; font-size:18px; font-weight:600; margin-bottom:16px">✅ 完成しました！プレビューで確認できます</p>
        <video id="previewVideo" class="preview-video" controls playsinline autoplay></video>
        <div style="margin-top:16px; display:flex; gap:10px; justify-content:center; flex-wrap:wrap;">
            <a class="download-btn" id="downloadBtn" href="#">⬇️ ダウンロード</a>
            <button class="btn-secondary" id="regenBtn" onclick="regenerate()">🔄 もう一度生成する</button>
        </div>
    </div>
    <div class="error-area" id="errorArea"></div>
</div>
<script>
let selectedFile = null;
let jobId = null;

function handleFileSelect(event) {
    selectedFile = event.target.files[0];
    if (selectedFile) {
        document.getElementById('fileName').textContent = selectedFile.name;
        document.getElementById('uploadBtn').disabled = false;
    }
}

async function uploadVideo() {
    if (!selectedFile) return;
    document.getElementById('uploadBtn').disabled = true;
    document.getElementById('progressArea').style.display = 'block';
    document.getElementById('downloadArea').style.display = 'none';
    document.getElementById('errorArea').style.display = 'none';
    const formData = new FormData();
    formData.append('video', selectedFile);
    const response = await fetch('/upload', { method: 'POST', body: formData });
    const data = await response.json();
    jobId = data.job_id;
    pollStatus();
}

function pollStatus() {
    const interval = setInterval(async () => {
        const response = await fetch('/status/' + jobId);
        const data = await response.json();
        document.getElementById('progressFill').style.width = data.progress + '%';
        document.getElementById('statusText').textContent = data.status;
        if (data.status === '完成') {
            clearInterval(interval);
            const v = document.getElementById('previewVideo');
            v.src = '/video/' + jobId + '?t=' + Date.now();   // 毎回最新を読む
            v.load();
            document.getElementById('downloadBtn').href = '/download/' + jobId;
            document.getElementById('downloadArea').style.display = 'block';
            document.getElementById('uploadBtn').disabled = false;
        } else if (data.status === 'エラー') {
            clearInterval(interval);
            document.getElementById('errorArea').style.display = 'block';
            document.getElementById('errorArea').textContent = 'エラーが発生しました: ' + data.error;
            document.getElementById('uploadBtn').disabled = false;
        }
    }, 2000);
}

async function regenerate() {
    document.getElementById('previewVideo').pause();
    document.getElementById('downloadArea').style.display = 'none';
    document.getElementById('progressArea').style.display = 'block';
    document.getElementById('statusText').textContent = '生成中';
    await fetch('/regenerate/' + jobId, { method: 'POST' });
    pollStatus();
}
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["video"]
    job_id = str(uuid.uuid4())[:8]
    video_path = f"{UPLOAD_FOLDER}/{job_id}_{file.filename}"
    file.save(video_path)
    jobs[job_id] = {"status": "処理中", "progress": 0}
    threading.Thread(target=process_video_auto, args=(job_id, video_path)).start()
    return jsonify({"job_id": job_id})

@app.route("/regenerate/<job_id>", methods=["POST"])
def regenerate(job_id):
    """同じ動画で別パターンを生成（編集なし・全自動）"""
    job = jobs.get(job_id)
    if not job or "cleaned" not in job:
        return jsonify({"error": "not ready"}), 400
    threading.Thread(target=regenerate_job, args=(job_id,)).start()
    return jsonify({"ok": True})

@app.route("/video/<job_id>")
def video(job_id):
    """完成動画をブラウザ内プレビュー用に返す（ダウンロードではなくインライン再生）"""
    job = jobs.get(job_id)
    if job and job.get("status") == "完成" and os.path.exists(job.get("output", "")):
        return send_file(job["output"], mimetype="video/mp4", conditional=True)
    return "Not found", 404

@app.route("/status/<job_id>")
def status(job_id):
    job = jobs.get(job_id, {"status": "不明", "progress": 0})
    return jsonify({"status": job.get("status"), "progress": job.get("progress", 0), "error": job.get("error")})

@app.route("/download/<job_id>")
def download(job_id):
    job = jobs.get(job_id)
    if job and job.get("status") == "完成":
        return send_file(job["output"], as_attachment=True, download_name="edited_video.mp4")
    return "Not found", 404

if __name__ == "__main__":
    # 5000番はmacOSのAirPlay受信機能と衝突する（403が返り真っ白になる）ため5001を使う
    app.run(debug=True, port=5001)
