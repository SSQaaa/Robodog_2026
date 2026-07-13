# -*- coding: utf-8 -*-
import queue
import subprocess
import threading
from pathlib import Path


AUDIO_DIR = Path(__file__).resolve().parent.parent / "wav"
APLAY_CMD = ["aplay", "-D", "plughw:2,0", "-c", "2", "-q"]
_AUDIO_QUEUE = queue.Queue()


def _audio_worker():
    while True:
        files = _AUDIO_QUEUE.get()
        try:
            result = subprocess.run(APLAY_CMD + [str(path) for path in files], check=False)
            if result.returncode != 0:
                print("[Audio] aplay failed, returncode={}".format(result.returncode))
        except FileNotFoundError:
            print("[Audio] aplay not found, skip announcement")
        finally:
            _AUDIO_QUEUE.task_done()


_AUDIO_THREAD = threading.Thread(target=_audio_worker, name="audio-announcer", daemon=True)
_AUDIO_THREAD.start()


def announce_dashboard(letter, state):
    """用 aplay 播报，例如：A区域、仪表盘显示、偏低、状态异常。"""
    letter = str(letter).upper()
    state = str(state)
    if letter not in ("A", "B", "C", "D") or state not in ("偏低", "偏高", "正常"):
        print("[Audio] skip invalid dashboard result: letter={} state={}".format(letter, state))
        return

    status = "状态正常" if state == "正常" else "状态异常"
    files = [
        AUDIO_DIR / (letter + "区域.wav"),
        AUDIO_DIR / "仪表盘显示.wav",
        AUDIO_DIR / (state + ".wav"),
        AUDIO_DIR / (status + ".wav"),
    ]
    print("[Audio] {}区域仪表盘显示{}，{}".format(letter, state, status))
    _AUDIO_QUEUE.put(files)
