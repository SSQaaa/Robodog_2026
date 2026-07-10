# -*- coding: utf-8 -*-
import subprocess
from pathlib import Path


AUDIO_DIR = Path(__file__).resolve().parent.parent / "wav"
APLAY_CMD = ["aplay", "-D", "plughw:2,0", "-c", "2", "-q"]


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
    try:
        subprocess.run(APLAY_CMD + [str(path) for path in files], check=False)
    except FileNotFoundError:
        print("[Audio] aplay not found, skip announcement")
