"""
AI編集エンジン（差し替え可能な層）。

ビジネスの「箱」からはこの process_video() だけを呼ぶ。
今はダミー（処理済みっぽい動画を返すだけ）。
あとで本物のAI編集エンジン（app.py 相当のロジック）をここに移植して差し替える。
"""
import os
import shutil
import subprocess


def _has_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


def process_video(input_path, output_path, plan="free"):
    """動画を編集して output_path に書き出す。

    引数:
        input_path : アップロードされた元動画のパス
        output_path: 完成動画の書き出し先パス
        plan       : ユーザーのプラン（"free" / "payg" / "monthly"）。
                     将来、プランごとに画質・長さ・透かしの有無などを変えるためのフック。

    返り値: output_path

    ※今はダミー実装。ffmpegがあれば「縦型・先頭30秒」に整形して“処理した感”を出し、
      無ければ単純コピーする。本物のAI編集は後でここに差し替える。
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if _has_ffmpeg():
        try:
            # ダミーの“編集”：先頭30秒を縦型(1080x1920)に整形。本物エンジン差し替え時にこの中身を置換する。
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", input_path, "-t", "30",
                    "-vf", "scale=1080:1920:force_original_aspect_ratio=increase,"
                           "crop=1080:1920",
                    "-c:a", "aac", "-c:v", "libx264", "-preset", "veryfast",
                    output_path,
                ],
                capture_output=True, check=True,
            )
            return output_path
        except Exception:
            pass  # ffmpegが失敗したらコピーにフォールバック

    shutil.copy(input_path, output_path)
    return output_path
