"""
AI編集エンジン（差し替え可能な層）。

ビジネスの「箱」(server.py)からはこの3つだけを使う:
  - process_video(input_path, output_path, plan)  … 全自動編集（一般プラン）
  - load_edit_data(output_path)                   … プロ編集用に保存したデータを読む
  - reedit(output_path, scenes=None, subtitle_texts=None) … プロの字幕/カット修正→再レンダリング

中身は app.py（テスト済みの本物パイプライン）の関数を再利用している。
"""
import os
import re
import json
from types import SimpleNamespace

import app as E  # 既存の本物エンジン（関数を再利用。__main__ガードがあるのでサーバは起動しない）

_FRAG = re.compile(r'^[぀-ゟ゠-ヿ]$')   # 単独かな1文字（断片）


def _stem(output_path):
    return os.path.splitext(output_path)[0]


def _data_path(output_path):
    return _stem(output_path) + "_edit.json"


# ---------------- パイプラインの部品 ----------------
def _transcribe_and_plan(input_path, stem):
    """文字起こし → AIで構成（見どころ選定）。再編集で使う素材も返す。"""
    audio = stem + "_audio.mp3"
    duration = E.get_video_duration(input_path)
    E.extract_audio(input_path, audio)
    transcript = E.transcribe_audio(audio)
    cleaned = E.clean_segments(transcript)
    structure = E.generate_structure(cleaned, duration)
    scenes, bgm_mood = E.parse_scenes(structure, cleaned, duration)
    words = [{"start": getattr(w, "start", None), "end": getattr(w, "end", None), "word": w.word}
             for w in transcript.words]
    return scenes, bgm_mood, words, cleaned, duration


def _word_objs(words):
    return [SimpleNamespace(**w) for w in words]


def _build_subtitles(words, scenes, cleaned):
    """編集後のシーンに合わせて字幕を作る（原音声マッピング→AI簡潔化→仕上げ）。"""
    cues = E.build_word_cues(_word_objs(words), scenes)
    reference = " ".join(s["text"].strip() for s in cleaned)
    subs = E.correct_segments(cues, reference=reference)
    for s in subs:
        s["text"] = E.strip_punct(E.sanitize_caption(s["text"]))
    return [s for s in subs if s["text"] and not _FRAG.match(s["text"])]


def _finalize(cut_path, subtitles, bgm_mood, stem, output_path):
    """カット済み動画に字幕を焼き込み、BGMをミックスして完成。"""
    srt = stem + "_subtitles.srt"
    subtitled = stem + "_subtitled.mp4"
    E.create_srt(subtitles, srt)
    E.burn_subtitles(cut_path, subtitles, subtitled)
    bgm_path, _ = E.get_bgm(bgm_mood)
    if bgm_path and E.mix_bgm(subtitled, bgm_path, output_path):
        if os.path.exists(subtitled):
            os.remove(subtitled)
    else:
        os.replace(subtitled, output_path)
    return output_path


def _save_data(output_path, data):
    with open(_data_path(output_path), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def load_edit_data(output_path):
    """プロ編集画面用：保存しておいた scenes / subtitles / bgm 等を読む（無ければNone）。"""
    p = _data_path(output_path)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def clean_caption(t):
    """字幕として表示する形に整える（句読点なし・余分記号なし）。"""
    return E.strip_punct(E.sanitize_caption(t or ""))


def list_segments(output_path):
    """プロ編集画面用：動画の“全文”（全セグメント）を返す。
    AIが採用した区間には selected=True を付ける（=おすすめに✅）。
    ユーザーはここから、AIが選ばなかった部分も含めて自由に取捨選択できる。
    """
    data = load_edit_data(output_path)
    if not data:
        return []
    ai = [(sc["start"], sc["end"]) for sc in data.get("scenes", [])]

    def in_ai(s, e):
        return any(s < ae and e > a_s for a_s, ae in ai)

    segs = []
    for seg in data.get("cleaned", []):
        segs.append({
            "start": seg["start"], "end": seg["end"],
            "text": clean_caption(seg["text"]),
            "selected": in_ai(seg["start"], seg["end"]),
        })
    return segs


def _tighten_scenes(scenes, words, pad=0.12, min_trim=0.25):
    """各シーンの前後の無音を、実際の発話（単語）の範囲まで詰める。
    → 間延びが消えてテンポUP、かつ境界が単語の頭/末尾に揃うので『マがあります』等の単語切れも防げる。
    詰めるのは無音が min_trim 秒より長い時だけ。pad 秒だけ余裕を残す。"""
    for sc in scenes:
        s, e = sc["start"], sc["end"]
        inside = [w for w in words
                  if w.get("start") is not None and w.get("end") is not None
                  and w["start"] >= s - 0.05 and w["start"] < e]
        if not inside:
            continue
        first = min(w["start"] for w in inside)
        last = max(w["end"] for w in inside)
        new_s = max(s, first - pad) if (first - s) > min_trim else s
        new_e = min(e, last + pad) if (e - last) > min_trim else e
        if new_e - new_s >= 0.4:
            sc["start"], sc["end"] = round(new_s, 3), round(new_e, 3)
    return scenes


# ---------------- 公開API ----------------
def process_video(input_path, output_path, plan="free"):
    """全自動編集（一般プラン）。完成動画を output_path に書き出す。

    プロプランでも同じ完成品を出しつつ、後で字幕/カットを修正できるよう素材を保存する。
    """
    stem = _stem(output_path)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    scenes, bgm_mood, words, cleaned, duration = _transcribe_and_plan(input_path, stem)

    # シーンが取れなかったら「失敗」として扱う（黙って未編集の素材を完成扱いにしない）
    if not scenes:
        raise RuntimeError("AIが編集構成を作れませんでした（素材が短すぎる/無音、またはAIの一時的な不調の可能性）")

    scenes = _tighten_scenes(scenes, words)      # 端の無音を発話まで詰める（テンポUP＋境界の単語切れ防止）
    cut_path = stem + "_cut.mp4"
    E.edit_video(input_path, scenes, cut_path)   # ここで scenes に out_start/out_end が付く
    subtitles = _build_subtitles(words, scenes, cleaned)
    _finalize(cut_path, subtitles, bgm_mood, stem, output_path)

    # プロ編集用に素材を保存（字幕テキスト修正・カット削除→再レンダリングに使う）
    _save_data(output_path, {
        "input_path": input_path,
        "bgm_mood": bgm_mood,
        "words": words,
        "cleaned": [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in cleaned],
        "scenes": [{"start": s["start"], "end": s["end"],
                    "transition": s.get("transition", "cut"),
                    "reason": s.get("reason", "")} for s in scenes],
        "subtitles": subtitles,
    })
    return output_path


def _find_scene(scenes, t):
    for sc in scenes:
        if sc["start"] - 0.05 <= t <= sc["end"] + 0.05:
            return sc
    return scenes[-1] if scenes else None


def reedit_textbased(output_path, kept_lines):
    """テキストベース編集（プロの魔法）：残した字幕から映像を切り直す。

    kept_lines = [{"text","orig_start","orig_end"}]（消した字幕は含めない）。
    「字幕を消す＝その区間の映像も消える」「字幕のテキストはユーザーの編集を保持」を実現する。
    """
    data = load_edit_data(output_path)
    if not data:
        raise RuntimeError("編集データが見つかりません")
    stem = _stem(output_path)

    # 残った字幕の“元動画の時刻”を時系列に並べ、隣接（間が短い）ものは1シーンにまとめる
    items = sorted([l for l in kept_lines
                    if l.get("orig_start") is not None and l.get("orig_end") is not None],
                   key=lambda x: x["orig_start"])
    if not items:
        raise RuntimeError("残す字幕がありません")

    MERGE_GAP = 0.6  # この秒数以内で隣り合う字幕は切らずに繋ぐ（不要なカットを増やさない）
    scenes = []
    for it in items:
        s, e = float(it["orig_start"]), float(it["orig_end"])
        if scenes and s - scenes[-1]["end"] <= MERGE_GAP:
            scenes[-1]["end"] = max(scenes[-1]["end"], e)
        else:
            scenes.append({"start": s, "end": e, "transition": "cut"})

    cut_path = stem + "_cut.mp4"
    E.edit_video(data["input_path"], scenes, cut_path)   # ここで out_start/out_end が付く

    # 残した区間を新タイムラインに再配置して字幕にする（ユーザーが直したテキストを保持・整形）。
    # テキストが空の区間は「映像は入れるが字幕は出さない」。
    subs = []
    for it in items:
        txt = clean_caption(it.get("text", ""))
        if not txt:
            continue
        s, e = float(it["orig_start"]), float(it["orig_end"])
        sc = _find_scene(scenes, s)
        if not sc:
            continue
        ns = max(sc["out_start"], min(s - sc["start"] + sc["out_start"], sc["out_end"]))
        ne = max(ns, min(e - sc["start"] + sc["out_start"], sc["out_end"]))
        subs.append({"start": ns, "end": ne, "text": txt,
                     "orig_start": s, "orig_end": e})
    subs.sort(key=lambda x: x["start"])

    _finalize(cut_path, subs, data["bgm_mood"], stem, output_path)
    data["scenes"] = scenes
    data["subtitles"] = subs
    _save_data(output_path, data)
    return output_path
