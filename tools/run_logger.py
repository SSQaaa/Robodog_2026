# -*- coding: utf-8 -*-
import csv
import json
import os
from datetime import datetime

from project_config import PROJECT_DIR


LOG_PATH = os.path.join(PROJECT_DIR, "logs", "run_timings.csv")


def _read_task1_cones():
    try:
        from task1 import PLAN_PATH, load_path_plan_data

        plan_data = load_path_plan_data(PLAN_PATH)
        return [cone.get("center") for cone in plan_data.get("cones_mm", [])]
    except Exception as exc:
        return "unavailable: {}".format(exc)


def append_run_log(task_seconds, total_seconds, status, errors=None, path=LOG_PATH, run_time=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    columns = [
        "run_time",
        "status",
        "task1_seconds",
        "task2_seconds",
        "task2_3_seconds",
        "task3_seconds",
        "total_seconds",
        "task1_cones_mm",
        "errors",
    ]
    row = {
        "run_time": run_time or datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": status,
        "task1_seconds": _format_seconds(task_seconds.get("task1")),
        "task2_seconds": _format_seconds(task_seconds.get("task2")),
        "task2_3_seconds": _format_seconds(task_seconds.get("task2_3")),
        "task3_seconds": _format_seconds(task_seconds.get("task3")),
        "total_seconds": _format_seconds(total_seconds),
        "task1_cones_mm": json.dumps(_read_task1_cones(), ensure_ascii=False),
        "errors": " | ".join(errors or []),
    }

    new_file = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="", encoding="utf-8-sig") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=columns)
        if new_file:
            writer.writeheader()
        writer.writerow(row)
    print("[Timing] run log saved: {}".format(path))


def _format_seconds(value):
    if value is None:
        return ""
    return "{:.3f}".format(float(value))
