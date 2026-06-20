from flask import Flask, request, jsonify, send_file, render_template_string
import os
import threading
import uuid
import subprocess
import json
import re
import difflib
from PIL import Image, ImageDraw, ImageFont
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

def _resolve_font():
    """字幕用の日本語フォントを環境に応じて探す（mac=ヒラギノ / Linux=Noto CJK）"""
    import glob
    candidates = [
        os.environ.get("SUBTITLE_FONT", ""),
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
                for j, part_text in enumerate(merged):
                    cleaned.append({"start": segment.start + j * part_duration, "end": segment.start + (j + 1) * part_duration, "text": part_text})
                continue
        cleaned.append({"start": segment.start, "end": segment.end, "text": text})
    return cleaned

def generate_structure(cleaned_segments, duration):
    segments_with_time = ""
    for seg in cleaned_segments:
        segments_with_time += f"[{seg['start']:.1f}秒〜{seg['end']:.1f}秒] {seg['text']}\n"
    message = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[{"role": "user", "content": f"""
あなたはTikTok・Instagram・YouTubeショートのプロ編集者です。
以下は{duration:.1f}秒の動画の音声テキストです。各行が1セグメントで、[開始秒〜終了秒]が付いています。

{segments_with_time}

この動画全体の中から、視聴者が最後まで見たくなる縦型ショート動画を構成してください。
目的は「長い動画を、テンポよくカットして“ちゃんと完結した”ショート動画にする」こと。

【最も大切なこと】
- 話の流れが最初から最後まで通っていること。起→展開→締めがあり、**途中でぶつ切りに終わらせない**
- **重要な場面・面白い場面・結末（オチ/まとめ）は必ず残す**。長さを気にして大事な部分を削らない
- 不要な部分（言い間違い、長い沈黙、冗長な繰り返し、どうでもいい雑談）は積極的に省いてテンポを上げる

【シーン選定の絶対ルール】（必ず守る）
- 各シーンの start / end は、上記セグメントの [開始秒〜終了秒] の値とそのまま一致させる。セグメントの途中の秒数で切らない
- セグメント1つ、または連続する複数セグメントをまとめて1シーンにする（区切りはセグメント境界のみ）
- 文章・話が完結しているシーンだけを選ぶ（言い切りの途中・文の途中で終わるシーンは選ばない）
- 前後のシーンが話の流れとして自然につながるように選ぶ（脈絡なく飛ばさない）
- 「えーと」「あの」「えー」などのフィラーで始まるセグメントはシーンの先頭にしない

編集の鉄則：
- 冒頭3秒で視聴者を引き込む（最もインパクトのある場面から始める）
- テンポを保つ（だらだらさせない）。1シーンは目安2〜6秒、強調したい所だけ長めにして緩急をつける
- シーンは時系列順に並べ、重複させない
- **最後は必ず話の締めくくり（まとめ・結論・オチ）で終わる**
- 全体の長さは30〜60秒を目安にする。ただし長さのために重要な場面や結末を削らない（完結を最優先し、必要なら短く/長くしてよい）

演出の指示：
- 動画全体に合うBGMのムード（bgm_mood）を1つ提案する（例：明るい / 感動的 / 緊張感 / おしゃれ / コミカル / 落ち着いた）
- 各シーンの切り替え方（transition）を指定する。場面や感情が大きく変わる箇所は "fade"、テンポよく繋ぐ箇所は "cut"

シーンの開始・終了秒数は必ず0以上{duration:.1f}以下にしてください。

以下のJSON形式のみで答えてください。
{{"bgm_mood": "動画全体に合うBGMのムード", "scenes": [{{"start": 開始秒数, "end": 終了秒数, "reason": "理由", "transition": "fade または cut"}}]}}
"""}]
    )
    text = message.content[0].text.replace("```json", "").replace("```", "").strip()
    return text

def remove_overlapping_scenes(scenes):
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

def mix_bgm(video_path, bgm_path, output_path, volume=0.15):
    result = subprocess.run([
        "ffmpeg", "-i", video_path, "-stream_loop", "-1", "-i", bgm_path,
        "-filter_complex", f"[1:a]volume={volume}[bg];[0:a][bg]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[a]",
        "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-shortest",
        output_path, "-y"
    ], capture_output=True)
    return result.returncode == 0

def correct_segments(segments):
    """字幕テキストを保守的に整える。ミスを増やさないことが最優先。
    入力・出力ともに [{start, end, text}] のリスト（タイミングは変えない）。"""
    if not segments:
        return []
    numbered = ""
    for i, seg in enumerate(segments):
        numbered += f"[{i}] {seg['text'].strip()}\n"
    try:
        message = anthropic_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            messages=[{"role": "user", "content": f"""
以下は動画の音声を自動認識した日本語字幕です。各行の [i] 番号はそのまま残し、テキストだけを読みやすく整えてください。

【最優先】ミスを増やさないこと。確信が持てない箇所は変えない。
- 明らかな誤字・脱字だけを直す
- 店名・地名・人名などの固有名詞は、確信が持てなければ変更しない（推測で別の漢字に置き換えない）
- 「えーと」「あの」「えー」「あ、」などの言い淀み・フィラーは削除する
- 句読点を自然に補う
- 話の意味を変えない。情報を足さない・要約しない
- [i] 番号付きの形式のまま出力する。説明や注釈は不要

テキスト：
{numbered}
"""}]
        )
        out = message.content[0].text.strip()
    except Exception:
        return [dict(s) for s in segments]  # 校正失敗時は原文のまま

    # [i] で分割して index→テキスト の対応表を作る（マーカー欠落でも後続を巻き込まない）
    parts = re.split(r'\[(\d+)\]', out)
    parsed = {}
    for k in range(1, len(parts) - 1, 2):
        text = parts[k + 1]
        text = re.sub(r'\[\d+\]', '', text)
        text = ' '.join(text.split())
        parsed[int(parts[k])] = text.strip()

    corrected = []
    for i, seg in enumerate(segments):
        text = parsed.get(i, "").strip() or seg["text"].strip()
        corrected.append({"start": seg["start"], "end": seg["end"], "text": text})
    return corrected

CUE_MAX_CHARS = 24   # 1キューの最大文字数
CUE_MAX_DUR = 5.0    # 1キューの最大表示秒数

def build_word_cues(words, scenes):
    """原音声の単語タイムスタンプを編集後タイムラインへマッピングして字幕を作る（v3方式：テキスト品質が高い）。
    各キューは所属シーンの編集後区間[out_start,out_end]にクランプし、ズレがシーンを越えて広がらないようにする。"""
    cues = []
    for scene in scenes:
        s, e = scene["start"], scene["end"]
        out_s, out_e = scene["out_start"], scene["out_end"]
        cur, c_start, c_end = "", None, None

        def emit():
            nonlocal cur, c_start, c_end
            text = cur.strip("、。 　")
            if text and c_start is not None:
                cs = max(out_s, min(c_start - s + out_s, out_e))   # シーン区間にクランプ（負値・はみ出し防止）
                ce = max(cs, min(c_end - s + out_s, out_e))
                cues.append({"start": cs, "end": ce, "text": text})
            cur, c_start, c_end = "", None, None

        for w in words:
            ws = getattr(w, "start", None)
            if ws is None or ws < s - 0.05 or ws >= e:
                continue
            token = w.word
            if token.strip("、。．！？!?・ 　") in FILLER_TOKENS:   # フィラー語は表示しない
                continue
            if c_start is None:
                c_start = ws
            cur += token
            c_end = max(getattr(w, "end", ws), c_start)
            if (token.rstrip().endswith(("。", "、", "．", "！", "？", "!", "?"))
                    or len(cur) >= CUE_MAX_CHARS or (c_end - c_start) >= CUE_MAX_DUR):
                emit()
        emit()

    cues.sort(key=lambda x: x["start"])
    result = []
    for c in cues:
        if c["end"] <= c["start"]:
            c["end"] = c["start"] + 0.4
        if result and c["start"] < result[-1]["end"]:
            result[-1]["end"] = c["start"]   # 重なり解消（同時に2つ表示しない）
        result.append(c)
    return result

def create_srt(segments, output_srt):
    def to_srt_time(seconds):
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

def wrap_text(text, font, max_width, draw):
    """字幕を読みやすく改行する。1行に収まればそのまま。
    収まらなければ「左右の長さが揃う中央付近」かつ「助詞・句読点など自然な位置」で2行に分ける。"""
    def width(s):
        b = draw.textbbox((0, 0), s, font=font)
        return b[2] - b[0]

    if width(text) <= max_width:
        return [text]

    n = len(text)
    target = n / 2
    best = None
    for i in range(1, n):
        if width(text[:i]) > max_width or width(text[i:]) > max_width:
            continue
        score = abs(i - target)            # 中央に近いほど良い（左右の文字数を揃える）
        prev, nxt = text[i - 1], text[i]
        if prev in "、。！？":
            score -= 8                     # 句読点の直後は最優先で改行
        elif prev in _BREAK_AFTER:
            score -= 4                     # 助詞の直後も自然
        if nxt in _BAD_LINE_HEAD:
            score += 8                     # 行頭に句読点・小書き文字が来るのは避ける
        if best is None or score < best[0]:
            best = (score, i)
    if best:
        i = best[1]
        return [text[:i], text[i:]]

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

def render_subtitle_png(text, w, h, path):
    """字幕1枚を透過PNGとして描画（下部中央・白文字＋黒フチ）"""
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font_size = 46
    try:
        font = ImageFont.truetype(SUBTITLE_FONT, font_size) if SUBTITLE_FONT else ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    lines = wrap_text(text, font, int(w * 0.86), draw)
    line_height = font_size + 12
    start_y = h - 140 - line_height * len(lines)
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        x = (w - (bbox[2] - bbox[0])) // 2
        try:
            draw.text((x, start_y), line, font=font, fill=(255, 255, 255, 255),
                      stroke_width=3, stroke_fill=(0, 0, 0, 255))
        except TypeError:
            for dx in (-2, 0, 2):
                for dy in (-2, 0, 2):
                    draw.text((x+dx, start_y+dy), line, font=font, fill=(0, 0, 0, 255))
            draw.text((x, start_y), line, font=font, fill=(255, 255, 255, 255))
        start_y += line_height
    img.save(path)

def burn_subtitles(input_path, subtitles, output_path):
    """字幕を透過PNGとしてoverlayで1パス合成する（全フレーム展開せず高速・無劣化）"""
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
        render_subtitle_png(sub["text"], w, h, png)
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

def process_phase_scenes(job_id, video_path):
    """段階1：文字起こし→候補シーンを用意。AIのおすすめ選択を付けて「シーン選択待ち」で停止。"""
    try:
        job = jobs[job_id]
        job.update(status="処理中", progress=10)
        audio_path = f"{OUTPUT_FOLDER}/{job_id}_audio.mp3"

        job["progress"] = 20
        duration = get_video_duration(video_path)
        extract_audio(video_path, audio_path)

        job["progress"] = 40
        transcript = transcribe_audio(audio_path)
        cleaned = clean_segments(transcript)

        job["progress"] = 65
        structure = generate_structure(cleaned, duration)
        ai_scenes, bgm_mood = parse_scenes(structure, cleaned, duration)

        # 字幕テキストを先に整える（確認画面で編集しやすいように）
        cleaned = correct_segments(cleaned)

        # 各候補に、その区間内の単語（表記＋タイムスタンプ）を保持（テキスト基準のカット用）
        words = getattr(transcript, "words", None) or []
        def seg_words(s):
            return [{"surface": getattr(w, "word", ""), "start": getattr(w, "start", 0.0), "end": getattr(w, "end", 0.0)}
                    for w in words if s["start"] - 0.01 <= getattr(w, "start", -1) < s["end"]]

        def in_ai(seg):
            return any(sc["start"] - 0.05 <= seg["start"] < sc["end"] for sc in ai_scenes)
        candidates = [{"start": s["start"], "end": s["end"], "text": s["text"],
                       "selected": in_ai(s), "words": seg_words(s)}
                      for s in cleaned]

        # 各候補シーンのサムネイル（中間地点のフレーム）を抽出 → 確認画面で表示
        job["progress"] = 68
        for i, c in enumerate(candidates):
            mid = (c["start"] + c["end"]) / 2
            subprocess.run(["ffmpeg", "-ss", f"{mid:.2f}", "-i", video_path, "-frames:v", "1",
                            "-vf", "scale=-2:120", "-q:v", "5",
                            f"{OUTPUT_FOLDER}/{job_id}_thumb_{i}.jpg", "-y"], capture_output=True)

        job["candidates"] = candidates
        job["video_path"] = video_path
        job["bgm_mood"] = bgm_mood
        job.update(status="確認待ち", progress=70)

    except Exception as e:
        jobs[job_id] = {"status": "エラー", "error": str(e)}

def _char_to_word(original, words):
    """元テキストの各文字が、どの単語(index)に対応するかを返す（単語の表記と文字単位で対応付け）"""
    wc, owner = [], []
    for wi, w in enumerate(words):
        for ch in w["surface"]:
            wc.append(ch)
            owner.append(wi)
    res = [None] * len(original)
    sm = difflib.SequenceMatcher(None, list(original), wc, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("equal", "replace"):
            for off in range(min(i2 - i1, j2 - j1)):
                res[i1 + off] = owner[j1 + off]
    last = None        # 前方向に補完
    for k in range(len(res)):
        if res[k] is None: res[k] = last
        else: last = res[k]
    nxt = None         # 後方向に補完
    for k in range(len(res) - 1, -1, -1):
        if res[k] is None: res[k] = nxt
        else: nxt = res[k]
    return res

def trimmed_range(seg, edited_text):
    """元テキスト(seg['text'])と編集後テキストを差分し、『削除された部分』を映像からトリムした
    [start, end] を返す。書き換え(誤字修正)はカット扱いにしない（=映像維持）。"""
    D = seg["text"]
    E = edited_text
    words = seg.get("words", [])
    full = (seg["start"], seg["end"])
    if not words or not E.strip():
        return full
    # 残る文字(equal/replace)のDインデックス範囲を求める（delete=削除のみカット対象）
    sm = difflib.SequenceMatcher(None, list(D), list(E), autojunk=False)
    kept = [i for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag in ("equal", "replace")
            for i in range(i1, i2)]
    if not kept:
        return full
    c2w = _char_to_word(D, words)
    wf, wl = c2w[min(kept)], c2w[max(kept)]
    if wf is None or wl is None:
        return full
    s = max(seg["start"], words[wf]["start"] - 0.08)
    e = min(seg["end"], words[wl]["end"] + 0.12)
    if e - s < 0.4:
        e = min(seg["end"], s + 0.4)
    return (s, e)

def process_phase_build(job_id, selected_indices, texts):
    """段階2：選ばれたシーン＋編集テキストで、編集→字幕→焼き込み→BGM→完成（1画面で確定）。
    テキストで削除した部分は映像もカット（テキスト基準の編集）。"""
    try:
        job = jobs[job_id]
        job.update(status="生成中", progress=75)
        cand = job["candidates"]
        job["last_selected"] = list(selected_indices)   # やり直し用に前回の編集内容を保持
        job["last_texts"] = list(texts)
        video_path = job["video_path"]
        bgm_mood = job.get("bgm_mood", "")
        output_path = f"{OUTPUT_FOLDER}/{job_id}_output.mp4"
        srt_path = f"{OUTPUT_FOLDER}/{job_id}_subtitles.srt"
        subtitled_path = f"{OUTPUT_FOLDER}/{job_id}_subtitled.mp4"
        final_path = f"{OUTPUT_FOLDER}/{job_id}_final.mp4"

        sel = sorted(i for i in selected_indices if 0 <= i < len(cand))
        if not sel:   # 何も選ばれなければAIのおすすめにフォールバック
            sel = [i for i, c in enumerate(cand) if c["selected"]] or [0]

        # 選択された各セグメント → 1シーン。編集テキストで削った分は映像もトリム。
        scenes, scene_text, prev_src_end = [], [], None
        for i in sel:
            seg = cand[i]
            edited = texts[i].strip() if i < len(texts) and isinstance(texts[i], str) else seg["text"]
            if not edited:                 # 全消し → そのシーンは使わない
                continue
            s, e = trimmed_range(seg, edited)
            trans = "fade" if (prev_src_end is None or s - prev_src_end > 2.0) else "cut"
            scenes.append({"start": s, "end": e, "transition": trans})
            scene_text.append(edited)
            prev_src_end = seg["end"]

        if not scenes:                     # 全部消された場合の保険
            seg = cand[sel[0]]
            scenes = [{"start": seg["start"], "end": seg["end"], "transition": "cut"}]
            scene_text = [seg["text"]]

        edit_video(video_path, scenes, output_path)   # 各sceneに out_start/out_end が付く

        # 字幕：各シーン（=トリム後クリップ）全体に、編集テキストを表示
        job["progress"] = 85
        subs = []
        for sc, text in zip(scenes, scene_text):
            subs.append({"start": sc["out_start"], "end": sc["out_end"], "text": text})
        result = []
        for c in sorted(subs, key=lambda x: x["start"]):
            if c["end"] <= c["start"]:
                c["end"] = c["start"] + 0.4
            if result and c["start"] < result[-1]["end"]:
                result[-1]["end"] = c["start"]
            result.append(c)
        subtitles = result

        create_srt(subtitles, srt_path)
        burn_subtitles(output_path, subtitles, subtitled_path)

        job["progress"] = 95
        bgm_path, _ = get_bgm(bgm_mood)
        if bgm_path and mix_bgm(subtitled_path, bgm_path, final_path):
            os.remove(subtitled_path)
        else:
            os.replace(subtitled_path, final_path)

        job.update(status="完成", progress=100, output=final_path)

    except Exception as e:
        jobs[job_id] = {"status": "エラー", "error": str(e)}

HTML = """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>動画AI編集ツール</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, sans-serif; background: #0f0f0f; color: #fff; min-height: 100vh; display: flex; align-items: center; justify-content: center; }
        .container { width: 100%; max-width: 480px; padding: 40px 20px; }
        h1 { font-size: 28px; font-weight: 700; margin-bottom: 8px; }
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
    <h1>✨ 動画AI編集</h1>
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
    <div class="scene-area" id="sceneArea">
        <h2>🎬 シーンを選んで字幕を直す</h2>
        <p class="hint">✅＝動画に入れるシーン。テキストの<b>一部を消すとその部分の映像もカット</b>されます（誤字を直すだけなら映像はそのまま）。</p>
        <div class="scene-list" id="sceneList"></div>
        <button class="btn" id="sceneBtn" onclick="submitScenes()">この内容で動画を作成</button>
    </div>
    <div class="download-area" id="downloadArea">
        <p style="color:#22c55e; font-size:18px; font-weight:600; margin-bottom:16px">✅ 完成しました！プレビューで確認できます</p>
        <video id="previewVideo" class="preview-video" controls playsinline></video>
        <div style="margin-top:16px; display:flex; gap:10px; justify-content:center; flex-wrap:wrap;">
            <a class="download-btn" id="downloadBtn" href="#">⬇️ ダウンロード</a>
            <button class="btn-secondary" id="reEditBtn" onclick="reEdit()">✏️ 編集に戻ってやり直す</button>
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
    document.getElementById('sceneArea').style.display = 'none';
    document.getElementById('sceneBtn').disabled = false;
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
        if (data.status === '確認待ち') {
            clearInterval(interval);
            showSceneSelector();
        } else if (data.status === '完成') {
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

function fmtTime(sec) {
    const m = Math.floor(sec / 60), s = Math.floor(sec % 60);
    return m + ':' + String(s).padStart(2, '0');
}

async function showSceneSelector() {
    const res = await fetch('/scenes/' + jobId);
    const data = await res.json();
    const list = document.getElementById('sceneList');
    list.innerHTML = '';
    data.scenes.forEach((sc, i) => {
        const row = document.createElement('div');
        row.className = 'scene-row' + (sc.selected ? ' on' : '');
        const cb = document.createElement('input');
        cb.type = 'checkbox';
        cb.checked = sc.selected;
        cb.dataset.idx = i;
        cb.onchange = () => row.classList.toggle('on', cb.checked);
        const img = document.createElement('img');
        img.className = 'thumb';
        img.src = '/thumb/' + jobId + '/' + i;
        img.loading = 'lazy';
        img.style.cursor = 'pointer';
        img.onclick = () => { cb.checked = !cb.checked; row.classList.toggle('on', cb.checked); };  // サムネクリックで選択切替
        const meta = document.createElement('div');
        meta.className = 'meta';
        meta.innerHTML = '<div class="t">' + fmtTime(sc.start) + ' 〜 ' + fmtTime(sc.end) + '</div>';
        const ta = document.createElement('textarea');
        ta.className = 'stxt';
        ta.rows = 1;
        ta.value = sc.text;
        ta.dataset.idx = i;
        meta.appendChild(ta);
        row.appendChild(cb);
        row.appendChild(img);
        row.appendChild(meta);
        list.appendChild(row);
    });
    document.getElementById('progressArea').style.display = 'none';
    document.getElementById('sceneArea').style.display = 'block';
}

async function submitScenes() {
    const rows = document.querySelectorAll('#sceneList .scene-row');
    const texts = new Array(rows.length).fill('');
    document.querySelectorAll('#sceneList .stxt').forEach(ta => { texts[parseInt(ta.dataset.idx)] = ta.value; });
    const selected = [];
    document.querySelectorAll('#sceneList input[type=checkbox]').forEach(b => { if (b.checked) selected.push(parseInt(b.dataset.idx)); });
    document.getElementById('sceneBtn').disabled = true;
    document.getElementById('sceneArea').style.display = 'none';
    document.getElementById('progressArea').style.display = 'block';
    await fetch('/select_scenes/' + jobId, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ selected, texts })
    });
    pollStatus();
}

function reEdit() {
    const v = document.getElementById('previewVideo');
    v.pause();
    document.getElementById('downloadArea').style.display = 'none';
    document.getElementById('sceneBtn').disabled = false;
    showSceneSelector();   // /scenes は前回の編集内容を反映して返す
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
    threading.Thread(target=process_phase_scenes, args=(job_id, video_path)).start()
    return jsonify({"job_id": job_id})

@app.route("/scenes/<job_id>")
def get_scenes(job_id):
    """候補シーン一覧を返す。やり直し時は前回の編集内容（選択・テキスト）を反映する。"""
    job = jobs.get(job_id, {})
    cand = job.get("candidates", [])
    last_texts = job.get("last_texts")
    last_selected = job.get("last_selected")
    out = []
    for i, c in enumerate(cand):
        text = last_texts[i] if (last_texts and i < len(last_texts) and isinstance(last_texts[i], str)) else c["text"]
        selected = (i in last_selected) if last_selected is not None else c["selected"]
        out.append({"start": c["start"], "end": c["end"], "text": text, "selected": selected})
    return jsonify({"scenes": out})

@app.route("/thumb/<job_id>/<int:idx>")
def thumb(job_id, idx):
    """シーン候補のサムネイル画像を返す"""
    path = f"{OUTPUT_FOLDER}/{job_id}_thumb_{idx}.jpg"
    if os.path.exists(path):
        return send_file(path, mimetype="image/jpeg")
    return "", 404

@app.route("/select_scenes/<job_id>", methods=["POST"])
def select_scenes(job_id):
    """選んだシーン(indices)と編集テキストを受け取り、編集→字幕→焼き込み→完成を開始する。
    完成後でも（候補が残っていれば）やり直し再生成を受け付ける。"""
    job = jobs.get(job_id)
    if not job or "candidates" not in job:
        return jsonify({"error": "not ready"}), 400
    data = request.get_json(force=True) or {}
    selected = data.get("selected", [])
    texts = data.get("texts", [])
    threading.Thread(target=process_phase_build, args=(job_id, selected, texts)).start()
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
