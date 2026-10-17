import datetime
import sys
import threading
from pathlib import Path

import cv2

SNAP_DIR = Path("./.log/captures")  # .log 已被 .gitignore 忽略，不污染仓库

_request = threading.Event()
_started = False


def _listen_stdin():
    """收到一整行就记一次请求。"""
    try:
        while sys.stdin.readline() != "":  # 回车即触发；EOF 则退出
            _request.set()
    except Exception:
        pass


def snapshot_requested() -> bool:
    """主循环每帧调用；有请求返回 True 并清零。首次调用时自动拉起监听线程。"""
    global _started
    if not _started:
        _started = True
        if not sys.stdin.isatty():
            print("[snapshot] stdin 不是终端，SSH 触发已禁用", file=sys.stderr)
        else:
            threading.Thread(
                target=_listen_stdin, daemon=True, name="snapshot-key"
            ).start()
    if _request.is_set():
        _request.clear()
        return True
    return False


def save_color_snapshot(image, out_dir=SNAP_DIR, prefix="color"):
    """保存一帧 BGR 图像为 jpg，返回路径；失败返回 None。"""
    if image is None:
        return None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = datetime.datetime.now().strftime(f"{prefix}_%Y%m%d_%H%M%S_%f")[:-3] + ".jpg"
    path = out_dir / name
    return str(path) if cv2.imwrite(str(path), image) else None
