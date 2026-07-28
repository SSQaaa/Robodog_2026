# -*- coding: utf-8 -*-
import traceback
import time
from datetime import datetime

import task1
import task2
import task2_3
from arm_control import reset_arm
from project_config import DEFAULT_DASHBOARD_STATUS
from task3 import Task3
from tools.motion import DogControl
from tools.run_logger import append_run_log
from tools.vision import resolve_dashboard_status


SHOW_TASK2_STREAM = True


def input_run_time():
    while True:
        run_time = input("请输入本次运行时间（月日时分，例如 07282206）：").strip()
        try:
            datetime.strptime(run_time, "%m%d%H%M")
        except ValueError:
            print("[Main] 时间格式无效，请输入8位月日时分，例如 07282206")
            continue
        return run_time


def get_task3_status(records):
    if not records:
        print("[Main] task2 did not return records, use default task3 status")
        return dict(DEFAULT_DASHBOARD_STATUS)
    try:
        status_by_letter = resolve_dashboard_status(records, default_status=DEFAULT_DASHBOARD_STATUS)
    except Exception:
        print("[Main] failed to resolve task2 records, use default task3 status")
        traceback.print_exc()
        return dict(DEFAULT_DASHBOARD_STATUS)
    if not status_by_letter:
        print("[Main] task2 status is empty, use default task3 status")
        return dict(DEFAULT_DASHBOARD_STATUS)
    return status_by_letter


def main():
    manual_run_time = input_run_time()
    dog = None
    task3_runner = None
    total_started_at = time.perf_counter()
    task_seconds = {}
    run_errors = []
    run_status = "failed"
    active_task = None
    task_started_at = None
    try:

        dog = DogControl()
        dog.stand_up()
        dog.close_continue()
        dog.stop()

        # 单独运行任务三的时候解注释
        # status_by_letter = dict(DEFAULT_DASHBOARD_STATUS)
        # print(f"[Main] task3-only status={status_by_letter}")

        start_yaw_deg = task2_3.read_current_yaw_deg()
        print("[Main] start_yaw_deg={:.3f}".format(start_yaw_deg))

        print("[Main] start task1")
        active_task = "task1"
        task_started_at = time.perf_counter()
        task1.run(dog)
        print("[Main] task1 finished, check yaw once")
        task2_3.rotate_to_relative_yaw_once(dog, start_yaw_deg - 90.0)
        task_seconds["task1"] = time.perf_counter() - task_started_at
        active_task = None
        time.sleep(2)
        # 换了个足端感觉没有变斜情况了
        # dog.move(last_time=0.18, vz=10000)

        print("[Main] start task2")
        active_task = "task2"
        task_started_at = time.perf_counter()
        try:
            records = task2.run(dog, show_stream=SHOW_TASK2_STREAM)
        except Exception as exc:
            print("[Main] task2 failed, use default task3 status")
            traceback.print_exc()
            run_errors.append("task2: {}".format(exc))
            records = None
        finally:
            task_seconds["task2"] = time.perf_counter() - task_started_at
            active_task = None
        status_by_letter = get_task3_status(records)
        print(f"[Main] task3 status={status_by_letter}")

        print("[Main] start task2_3 bridge")
        active_task = "task2_3"
        task_started_at = time.perf_counter()
        task2_3.run(dog, start_yaw_deg=start_yaw_deg)
        task_seconds["task2_3"] = time.perf_counter() - task_started_at
        active_task = None
        time.sleep(2)
        dog.revolve_180()
        time.sleep(2)

        print("[Main] reset arm before task3")
        active_task = "task3"
        task_started_at = time.perf_counter()
        reset_arm()
        task3_runner = Task3(status_dict=status_by_letter, dog=dog)
        print("[Main] start task3")
        task3_runner.start()
        task3_runner.run()
        task_seconds["task3"] = time.perf_counter() - task_started_at
        active_task = None

        print("[Main] all tasks finished")
        run_status = "completed_with_errors" if run_errors else "completed"
    except Exception as exc:
        print("[Main] failed, stopping robot")
        traceback.print_exc()
        run_errors.append("main: {}".format(exc))
        if dog is not None:
            dog.stop()
        raise
    finally:
        if active_task is not None and active_task not in task_seconds:
            task_seconds[active_task] = time.perf_counter() - task_started_at
        if task3_runner is not None:
            try:
                task3_runner.close()
            except Exception:
                print("[Main] failed to close task3 runner")
                traceback.print_exc()
        if dog is not None:
            try:
                dog.stop()
                dog.close()
            except Exception:
                print("[Main] failed to close dog control")
                traceback.print_exc()
        try:
            append_run_log(
                task_seconds=task_seconds,
                total_seconds=time.perf_counter() - total_started_at,
                status=run_status,
                errors=run_errors,
                run_time=manual_run_time,
            )
        except Exception:
            print("[Timing] failed to save run log")
            traceback.print_exc()


if __name__ == "__main__":
    main()
