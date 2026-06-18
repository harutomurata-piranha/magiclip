import openai
import anthropic
import subprocess
import os
import json
import re
from PIL import Image, ImageDraw, ImageFont
import shutil
from dotenv import load_dotenv

load_dotenv()

# APIキー設定（.envから読み込み）
openai_client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

def get_video_duration(video_path):
    result = subprocess.run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ], capture_output=True, text=True)
    duration = float(result.stdout.strip())
    print(f"📏 動画の長さ：{duration:.1f}秒")
    return duration

def extract_audio(video_path, audio_path):
    print("🎬 音声を抽出中...")
    subprocess.run([
        "ffmpeg", "-i", video_path,
        "-vn", "-ar", "44100", "-ac", "2", "-b:a", "128k",
        audio_path, "-y"
    ])
    print("✅ 音声抽出完了")

def transcribe_audio(audio_path):
    print("🎤 音声をテキスト化中...")
    audio_file = open(audio_path, "rb")
    transcript = openai_client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        language="ja",
        response_format="verbose_json",
        timestamp_granularities=["segment", "word"]
    )
    print("✅ テキスト化完了")
    return transcript

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
    # フィラー・相槌を除いて実質的な中身が残らないセグメントは除外（例：「はい。」）
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
                    cleaned.append({
                        "start": segment.start + j * part_duration,
                        "end": segment.start + (j + 1) * part_duration,
                        "text": part_text
                    })
                continue

        cleaned.append({
            "start": segment.start,
            "end": segment.end,
            "text": text
        })

    return cleaned

def generate_structure(fixed_segments, duration):
    print("🤖 ストーリー構成を生成中...")

    segments_with_time = ""
    for seg in fixed_segments:
        segments_with_time += f"[{seg['start']:.1f}秒〜{seg['end']:.1f}秒] {seg['text']}\n"

    message = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[
            {
                "role": "user",
                "content": f"""
あなたはTikTok・Instagram・YouTubeショートのプロ編集者です。
以下は{duration:.1f}秒の動画の音声テキストです。各行が1セグメントで、[開始秒〜終了秒]が付いています。

{segments_with_time}

この動画全体の中から、視聴者が最後まで見たくなる縦型ショート動画の構成を作ってください。

【シーン選定の絶対ルール】（最優先・必ず守る）
- 各シーンの start / end は、上記セグメントの [開始秒〜終了秒] の値とそのまま一致させる。セグメントの途中の秒数で切らない
- セグメント1つ、または連続する複数セグメントをまとめて1シーンにする（区切りはセグメント境界のみ）
- 文章・話が完結しているシーンだけを選ぶ（言い切りの途中・文の途中で終わるシーンは選ばない）
- 前後のシーンが話の流れとして自然につながるように選ぶ（脈絡なく飛ばさない）
- 「えーと」「あの」「えー」などのフィラーで始まるセグメントはシーンの先頭にしない（避ける）

編集の鉄則：
- 動画全体から最も面白い・価値あるシーンを選ぶ
- 冒頭3秒で視聴者を引き込む（最もインパクトのある瞬間から始める）
- 1シーンは2〜5秒以内に収める（短い場合は連続セグメントをまとめる）
- シーンは時系列順に並べる（前のシーンの終了秒数より後のシーンの開始秒数が大きくなるようにする）
- シーン同士が重複しないようにする
- 重要なポイントのシーンだけ少し長めに見せる（緩急をつける）
- 締めはテンポを落として印象を残す
- 全体の長さは20〜40秒に収める

演出の指示：
- 動画全体の雰囲気に合うBGMのムード（bgm_mood）を1つ提案する（例：明るい / 感動的 / 緊張感 / おしゃれ / コミカル / 落ち着いた）
- 各シーンの切り替え方（transition）を指定する。場面や感情が大きく変わる箇所は "fade"（フェード）、テンポよく繋ぐ箇所は "cut"（カット）

シーンの開始・終了秒数は必ず0以上{duration:.1f}以下にしてください。

以下のJSON形式のみで答えてください。他の文章は不要です。
{{
    "bgm_mood": "動画全体に合うBGMのムードを一言で",
    "scenes": [
        {{"start": 開始秒数, "end": 終了秒数, "reason": "このシーンを選んだ理由", "transition": "fade または cut"}},
        {{"start": 開始秒数, "end": 終了秒数, "reason": "このシーンを選んだ理由", "transition": "fade または cut"}}
    ]
}}
"""
            }
        ]
    )
    print("✅ ストーリー構成完了")
    text = message.content[0].text
    text = text.replace("```json", "").replace("```", "").strip()
    return text

def remove_overlapping_scenes(scenes):
    scenes = sorted(scenes, key=lambda x: x["start"])
    cleaned = []
    last_end = -1

    for scene in scenes:
        if scene["start"] >= last_end:
            cleaned.append(scene)
            last_end = scene["end"]
        else:
            if scene["end"] > last_end:
                scene["start"] = last_end
                cleaned.append(scene)
                last_end = scene["end"]

    return cleaned

FADE_DUR = 0.15      # フェードイン・アウトの長さ（秒）※境界の黒フレームを抑えるため短めに設定

def build_fade_filters(scene_duration):
    """シーンの長さに合わせたフェードイン・アウトのフィルタ文字列を返す"""
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
    bgm_mood, raw = "", []
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
        print("⚠️ 有効なシーンが無いためフォールバック構成を使用します")
        valid = fallback_scenes(cleaned_segments, duration)
    return valid, bgm_mood

def edit_video(video_path, scenes, output_path):
    print("✂️ 動画を編集中...")

    temp_files = []
    timeline = 0.0
    for i, scene in enumerate(scenes):
        scene_duration = scene["end"] - scene["start"]
        temp_file = f"temp_scene_{i}.mp4"
        # -ss / -to は -i の前（入力シーク）に置く。フェードを切り出し区間の先頭=0秒基準で効かせるため
        cmd = [
            "ffmpeg",
            "-ss", str(scene["start"]),
            "-to", str(scene["end"]),
            "-i", video_path,
        ]
        transition = scene.get("transition", "cut")
        if transition == "fade":
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

        print(f"  ✅ シーン{i+1}：{scene['start']}秒〜{scene['end']}秒 [{transition}]")

    with open("concat_list.txt", "w") as f:
        for temp_file in temp_files:
            f.write(f"file '{temp_file}'\n")

    subprocess.run([
        "ffmpeg", "-f", "concat", "-safe", "0",
        "-i", "concat_list.txt",
        "-vf", "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2",
        "-c:v", "libx264", "-c:a", "aac",
        output_path, "-y"
    ], capture_output=True)

    for temp_file in temp_files:
        os.remove(temp_file)
    os.remove("concat_list.txt")

    print(f"✅ 動画編集完了：{output_path}")
    return scenes

# ===== BGM（Pixabay等のCC0音源 / クレジット表記不要 / bgm/ に手動配置）=====
# Pixabay( https://pixabay.com/music/ )などからCC0のmp3をダウンロードし、
# 下記カテゴリ名で bgm/ フォルダに置いてください（例：bgm/bright.mp3）。
#   bright / emotional / tense / stylish / comical  ＋ 任意の default.mp3（フォールバック）
BGM_FOLDER = "bgm"
MOOD_KEYWORDS = {
    "bright":    ["明る", "楽し", "ポップ", "元気", "ハッピー", "happy", "bright", "fun", "upbeat"],
    "emotional": ["感動", "エモ", "壮大", "感情", "泣", "epic", "emotional", "inspir", "moving"],
    "tense":     ["緊張", "シリアス", "ドラマ", "迫力", "tense", "serious", "drama", "suspense"],
    "stylish":   ["おしゃれ", "オシャレ", "chill", "リラックス", "落ち着", "穏やか", "lofi", "lo-fi", "stylish", "calm", "relax"],
    "comical":   ["コミカル", "ユーモア", "面白", "ふざけ", "コメディ", "funny", "comic", "comical", "quirky"],
}

def bgm_category(mood):
    """Claudeが提案したBGMムード（自由記述）をカテゴリに対応づける"""
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
            print(f"🎵 BGM：{name}.mp3（ムード:{mood or '未指定'} → {category}）")
            return path, name
    print(f"ℹ️ bgm/{category}.mp3 も bgm/default.mp3 も見つかりません")
    return None, None

def mix_bgm(video_path, bgm_path, output_path, volume=0.15):
    """元の音声を保ったまま、低音量のBGMをループ・ミックスする"""
    print("🎚️ BGMをミックス中...")
    result = subprocess.run([
        "ffmpeg",
        "-i", video_path,
        "-stream_loop", "-1", "-i", bgm_path,
        "-filter_complex",
        f"[1:a]volume={volume}[bg];[0:a][bg]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[a]",
        "-map", "0:v", "-map", "[a]",
        "-c:v", "copy", "-c:a", "aac", "-shortest",
        output_path, "-y"
    ], capture_output=True)
    if result.returncode != 0:
        print("⚠️ BGMミックスに失敗したためBGMなしの動画を使用します")
        return False
    print(f"✅ BGMミックス完了：{output_path}")
    return True

def correct_segments(segments):
    """字幕テキストを保守的に整える。ミスを増やさないことが最優先。
    入力・出力ともに [{start, end, text}]（タイミングは変えない）。"""
    if not segments:
        return []
    print("✏️ 字幕テキストを校正中...")
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
        print("⚠️ 校正に失敗したため原文のまま使用します")
        return [dict(s) for s in segments]

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
    print("✅ 校正完了")
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
            result[-1]["end"] = c["start"]   # 重なりを解消（同時に2つ表示しない）
        result.append(c)
    return result

def create_srt(segments, output_srt):
    print("📝 字幕ファイルを生成中...")

    def to_srt_time(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    with open(output_srt, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments):
            f.write(f"{i+1}\n")
            f.write(f"{to_srt_time(segment['start'])} --> {to_srt_time(segment['end'])}\n")
            f.write(f"{segment['text']}\n\n")
    print("✅ 字幕ファイル生成完了")

def wrap_text(text, font, max_width, draw):
    lines = []
    current_line = ""

    for char in text:
        test_line = current_line + char
        bbox = draw.textbbox((0, 0), test_line, font=font)
        text_w = bbox[2] - bbox[0]

        if text_w <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
            current_line = char

    if current_line:
        lines.append(current_line)

    return lines

def render_subtitle_png(text, w, h, path):
    """字幕1枚を透過PNGとして描画（下部中央・白文字＋黒フチ）"""
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font_size = 46
    try:
        font = ImageFont.truetype("/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc", font_size)
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
        print(f"✅ 完成（字幕なし）：{output_path}")
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
        png = f"_sub_{i}.png"
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
    print(f"✅ 完成：{output_path}")

# メイン処理
video_path = "IMG_0798.MOV"
audio_path = "temp_audio.mp3"
srt_path = "subtitles.srt"
output_path = "output.mp4"
subtitled_path = "subtitled.mp4"
final_path = "final.mp4"

# 1. 動画の長さを自動取得
duration = get_video_duration(video_path)

# 2. 音声抽出
extract_audio(video_path, audio_path)

# 3. テキスト化（シーン選定用）
transcript = transcribe_audio(audio_path)
print(f"\n📝 元テキスト：\n{transcript.text}\n")

# 4. セグメント整理（聞き取れるシーンだけに絞り込み）
cleaned_segments = clean_segments(transcript)
print(f"🧹 整理後セグメント数：{len(cleaned_segments)}")

# 5. ストーリー構成（失敗・空でもフォールバックで必ずシーンを得る）
structure = generate_structure(cleaned_segments, duration)
print(f"\n🎯 ストーリー構成：\n{structure}\n")
scenes, bgm_mood = parse_scenes(structure, cleaned_segments, duration)

# 6. 動画編集
edit_video(video_path, scenes, output_path)

# 7. 字幕：原音声の単語タイムスタンプを編集後タイムラインへマッピング（v3方式：テキスト品質が高い）
cues = build_word_cues(transcript.words, scenes)
subtitles = correct_segments(cues)
create_srt(subtitles, srt_path)

# 8. 字幕を映像に焼き込む（透過PNG＋overlayの1パス）
print("🔤 字幕を焼き込み中...")
burn_subtitles(output_path, subtitles, subtitled_path)

# 9. BGMを自動ミックス
print(f"🎼 BGMムード：{bgm_mood or '（未指定）'}")
bgm_path, category = get_bgm(bgm_mood)
if bgm_path and mix_bgm(subtitled_path, bgm_path, final_path):
    os.remove(subtitled_path)
    print(f"🎵 BGMをミックスしました：{os.path.basename(bgm_path)}（CC0 / クレジット表記不要）")
else:
    os.replace(subtitled_path, final_path)
    print("ℹ️ BGMなしで書き出しました（bgm/フォルダにCC0音源を置くと自動でミックスされます）")

print(f"🎉 完成：{final_path}")