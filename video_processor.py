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
import subprocess
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
    subs = [s for s in subs if s["text"] and not _FRAG.match(s["text"])]
    return E.dedup_consecutive_subs(subs)   # 校正後の連続重複も除去


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
    # 透かし「✦ MagiClip」を右下に焼き込む（仕上げ）
    wm = stem + "_wm_final.mp4"
    if E.add_watermark(output_path, wm):
        os.replace(wm, output_path)
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


def editor_text(t):
    """編集画面の初期テキスト用：clean_caption に加え、区切られたフィラー語だけ安全に除去する
    （語の途中は壊さない＝空白で分かれた『はい』『えーと』等のトークンのみ落とす）。"""
    toks = [w for w in clean_caption(t).split(" ") if w and w not in E.FILLER_TOKENS]
    return " ".join(toks)


def list_segments(output_path):
    """プロ編集画面用：動画の“全文”（全セグメント＝シーン候補）を返す。
    各シーン: {i, start, end, text(編集可), selected(AI採用=ON)}。
    AIが選ばなかったシーンも含めて全部返す（ユーザーが取捨選択・編集できる）。
    前回の編集(seg_keep / seg_edits)があれば復元する。
    """
    data = load_edit_data(output_path)
    if not data:
        return []
    cleaned = data.get("cleaned", [])
    ai = [(sc["start"], sc["end"]) for sc in data.get("scenes", [])]
    keep = data.get("seg_keep")
    edits = data.get("seg_edits", {})
    if keep is not None and any(int(k) >= len(cleaned) for k in keep):
        keep, edits = None, {}      # 不整合な保存データは無視

    def in_ai(s, e):
        return any(s < ae and e > a_s for a_s, ae in ai)

    out = []
    for i, seg in enumerate(cleaned):
        sel = (i in keep) if keep is not None else in_ai(seg["start"], seg["end"])
        out.append({
            "i": i,
            "start": seg["start"], "end": seg["end"],
            "text": edits.get(str(i), editor_text(seg["text"])),
            "selected": sel,
        })
    return out


def segment_thumbnail(output_path, idx):
    """シーン候補のサムネイル（区間の中間フレーム）を元動画から作って返す（無ければ生成）。"""
    data = load_edit_data(output_path)
    if not data:
        return None
    cleaned = data.get("cleaned", [])
    if idx < 0 or idx >= len(cleaned):
        return None
    thumb = _stem(output_path) + f"_thumb_{idx}.jpg"
    if not os.path.exists(thumb):
        seg = cleaned[idx]
        mid = (seg["start"] + seg["end"]) / 2
        subprocess.run(["ffmpeg", "-ss", f"{mid:.2f}", "-i", data["input_path"], "-frames:v", "1",
                        "-vf", "scale=-2:160", "-q:v", "5", thumb, "-y"], capture_output=True)
    return thumb if os.path.exists(thumb) else None


def _nospace(t):
    return clean_caption(t).replace(" ", "")


def _seg_kept_ranges(s, e, edited, words):
    """1シーン内で「編集後テキストに残っている単語」の時間範囲を返す。
    文字を削った分だけ映像を短くするための核。誤字修正（長さがほぼ同じ）はカットしない。"""
    inseg = [w for w in words if w.get("start") is not None and s - 0.05 <= w["start"] < e]
    orig = _nospace("".join(w["word"] for w in inseg))
    ed = _nospace(edited)
    # 元とほぼ同じ長さ＝誤字修正だけ → カットせずシーン全体を残す
    if not inseg or len(ed) >= len(orig) - 1:
        return [(s, e)]
    # 短くなった → 残っている単語だけ時間範囲を拾う（消した単語の所はカット）
    ranges, pos = [], 0
    for w in inseg:
        wt = _nospace(w["word"])
        if not wt:
            continue
        idx = ed.find(wt, pos)
        if idx != -1:
            ranges.append([w["start"], w.get("end") or w["start"]])
            pos = idx + len(wt)
    if not ranges:
        return [(s, e)]   # 対応が取れない時は安全側で全体を残す
    merged = [ranges[0]]
    for a, b in ranges[1:]:
        if a - merged[-1][1] <= 0.15:        # 隣り合う単語は繋ぐ
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def reedit_segments(output_path, kept):
    """シーン単位の再編集（言葉起点）。選んだシーン＋編集テキストで映像を切り直す。
    kept = [{"i","text","start","end"}]。
    『シーンを外す＝その映像をカット／文字を一部消す＝その単語ぶんの映像も短くなる／文字を直す＝字幕も直る』。"""
    data = load_edit_data(output_path)
    if not data:
        raise RuntimeError("編集データが見つかりません")
    stem = _stem(output_path)
    words = data.get("words", [])
    items = sorted([s for s in kept if s.get("start") is not None
                    and clean_caption(s.get("text", ""))], key=lambda x: x["start"])
    if not items:
        raise RuntimeError("残すシーンがありません")

    # 各シーンの「残す時間範囲」を編集テキストから割り出す（文字を削った分は範囲から除く＝カット）
    flat = []   # (start, end, item_index)
    for k, it in enumerate(items):
        for a, b in _seg_kept_ranges(float(it["start"]), float(it["end"]), it["text"], words):
            flat.append((a, b, k))
    flat.sort(key=lambda x: x[0])

    # 連続範囲は1シーンに繋ぎ、離れた所はfade。範囲→出力シーンの対応を記録
    scenes, prev_end = [], None
    flat_scene = []
    for a, b, k in flat:
        if scenes and prev_end is not None and a - prev_end < 0.3:
            scenes[-1]["end"] = b
        else:
            trans = "fade" if (prev_end is not None and a - prev_end > 2.0) else "cut"
            scenes.append({"start": a, "end": b, "transition": trans})
        flat_scene.append(len(scenes) - 1)
        prev_end = b

    E.edit_video(data["input_path"], scenes, stem + "_cut.mp4")   # out_start/out_end が付く

    # 字幕：各シーン(item)の編集テキストを、最初の残存範囲の編集後タイムラインに配置
    first = {}    # item_index -> (scene_idx, orig_start, orig_end)
    for (a, b, k), scn in zip(flat, flat_scene):
        if k not in first:
            first[k] = [scn, a, b]
        else:
            first[k][2] = b
    subs = []
    for k, it in enumerate(items):
        if k not in first:
            continue
        scn, a, b = first[k]
        sc = scenes[scn]
        cs = max(sc["out_start"], min(a - sc["start"] + sc["out_start"], sc["out_end"]))
        ce = max(cs, min(b - sc["start"] + sc["out_start"], sc["out_end"]))
        subs.append({"start": cs, "end": ce, "text": clean_caption(it["text"])})
    subs.sort(key=lambda x: x["start"])
    res = []
    for c in subs:
        if c["end"] <= c["start"]:
            c["end"] = c["start"] + 0.4
        if res and c["start"] < res[-1]["end"]:
            res[-1]["end"] = c["start"]
        res.append(c)
    res = E.dedup_consecutive_subs(res)

    _finalize(stem + "_cut.mp4", res, data["bgm_mood"], stem, output_path)
    data["scenes"] = [{"start": s["start"], "end": s["end"], "transition": s.get("transition", "cut")} for s in scenes]
    data["subtitles"] = res
    data["seg_keep"] = [it["i"] for it in items if "i" in it]
    data["seg_edits"] = {str(it["i"]): it["text"] for it in items if "i" in it and it.get("text")}
    _save_data(output_path, data)
    return output_path


def words_of(data):
    return data.get("words", [])


# チップ（編集の単位）の区切り：句読点＋文節末になりやすい助詞。語の途中で割れにくい文節サイズにする
_CHIP_BREAK = set("、。！？はがをにへとでもねよわ")


def _group_chips(words):
    """whisperの細かい単語トークンを"文節サイズ"のチップにまとめる（郵/便/局→郵便局ですね）。
    各チップ: {text, start, end, br}（br=直前と間が空く＝改行の目印）。"""
    chips, cur, cstart, cend, prev_end, pending_br = [], "", None, None, None, False

    def flush():
        nonlocal cur, cstart, cend, pending_br
        if cur.strip():
            chips.append({"text": cur, "start": cstart, "end": cend, "br": pending_br})
            pending_br = False
        cur, cstart, cend = "", None, None

    for w in words:
        s = w.get("start")
        if s is None:
            continue
        e = w.get("end") or s
        if cstart is not None and prev_end is not None and (s - prev_end) >= 0.4:
            flush()
            pending_br = True          # 間＝次チップの前で改行
        if cstart is None:
            cstart = s
        cur += w["word"]
        cend = e
        prev_end = e
        last = cur.rstrip("　 ")[-1:]
        if (last in _CHIP_BREAK and len(cur) >= 2) or len(cur) >= 10:
            flush()
    flush()
    return chips


def list_words(output_path):
    """プロ編集（言葉起点）画面用：動画の“全文”を文節チップで返す。
    - selected: AIが採用した区間にあるチップ（既定でON）。未採用も含め全チップを返す
    - text: 表示/編集する文字（過去の編集があればそれを反映）
    - br: 改行の目印（読みやすさ）
    ユーザーはチップを消す→その映像も短くなる、未採用のチップを足す、誤字を直す、ができる。
    """
    data = load_edit_data(output_path)
    if not data:
        return []
    ai = [(sc["start"], sc["end"]) for sc in data.get("scenes", [])]
    edits = data.get("word_edits", {})          # {chip_index: 直したテキスト}
    keep = data.get("word_keep")                # 前回残したchip index（無ければAI採用で初期化）

    def in_ai(s, e):
        return any(s < ae and e > a_s for a_s, ae in ai)

    chips = _group_chips(words_of(data))
    # 旧形式（単語index）など、チップ数と整合しない保存データは無視してAI採用にフォールバック
    if keep is not None and any(int(k) >= len(chips) for k in keep):
        keep, edits = None, {}

    out = []
    for i, ch in enumerate(chips):
        sel = (i in keep) if keep is not None else in_ai(ch["start"], ch["end"])
        out.append({
            "i": i,
            "text": edits.get(str(i), ch["text"]),
            "start": ch["start"], "end": ch["end"],
            "selected": sel, "br": ch["br"],
        })
    return out


def reedit_words(output_path, kept):
    """言葉起点の再編集：残した単語だけから映像を切り直す。
    kept = [{"i","text","start","end"}]（消した単語は含めない）。
    『単語を消す＝その区間の映像も消える／文字を直す＝字幕も直る』を実現する。"""
    data = load_edit_data(output_path)
    if not data:
        raise RuntimeError("編集データが見つかりません")
    stem = _stem(output_path)
    items = sorted([w for w in kept if w.get("start") is not None and w.get("end") is not None],
                   key=lambda w: w["start"])
    if not items:
        raise RuntimeError("残す言葉がありません")

    # 残した単語を時系列に並べ、隣り合う（間がごく短い）ものは1シーンに繋ぐ。
    # 消した単語の所は間が空く→そこでシーンが分かれる＝映像がカットされる。
    GAP = 0.12
    scenes = []
    for w in items:
        s, e = float(w["start"]), float(w["end"])
        if scenes and s - scenes[-1]["end"] <= GAP:
            scenes[-1]["end"] = max(scenes[-1]["end"], e)
        else:
            scenes.append({"start": s, "end": e, "transition": "cut"})

    cut_path = stem + "_cut.mp4"
    E.edit_video(data["input_path"], scenes, cut_path)   # out_start/out_end が付く

    # 字幕：残した単語（編集後テキスト）を build_word_cues で読みやすい塊にまとめる
    wobjs = [SimpleNamespace(start=w["start"], end=w["end"], word=w["text"]) for w in items]
    subs = E.build_word_cues(wobjs, scenes)
    for s in subs:
        s["text"] = clean_caption(s["text"])
    subs = [s for s in subs if s["text"] and not _FRAG.match(s["text"])]
    subs = E.dedup_consecutive_subs(subs)

    _finalize(cut_path, subs, data["bgm_mood"], stem, output_path)
    # 次回の編集画面で状態を復元できるよう保存
    data["scenes"] = [{"start": s["start"], "end": s["end"], "transition": "cut"} for s in scenes]
    data["subtitles"] = subs
    data["word_keep"] = [w["i"] for w in items if "i" in w]
    data["word_edits"] = {str(w["i"]): w["text"] for w in items
                          if "i" in w and w.get("text")}
    _save_data(output_path, data)
    return output_path


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
