# -*- coding: utf-8 -*-
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
# TODO：前两次yaw校正改成2度。终点的设定，平移多就设为4.35，平移少就设置为4.2。C回来走1s。任务一结束的向前时间？
import task1
import task2
import task2_3
from arm_control import reset_arm
from project_config import DEFAULT_DASHBOARD_STATUS
from task3 import Task3
from tools.motion import DogControl
from tools.run_logger import append_run_log
from tools.vision_manager import VisionManager
from tools.world_pose import close_pose_reader, correct_yaw


SHOW_TASK2_STREAM = False


@contextmanager
def measure_task(task_seconds, name):
    started_at = time.perf_counter()
    try:
        yield
    finally:
        task_seconds[name] = time.perf_counter() - started_at


def input_run_time():
    while True:
        run_time = input("请输入本次运行时间（月日时分，例如 07282206）：").strip()
        try:
            datetime.strptime(run_time, "%m%d%H%M")
            return run_time
        except ValueError:
            print("[Main] 时间格式无效，请输入 8 位月日时分，例如 07282206")


def get_task3_status(records):
    _ = records
    print("[Main] use project_config DEFAULT_DASHBOARD_STATUS for task3")
    return dict(DEFAULT_DASHBOARD_STATUS)


def run_task2(dog, detector, run_errors):
    try:
        return task2.run(dog, show_stream=SHOW_TASK2_STREAM, detector=detector)
    except Exception as exc:
        print("[Main] task2 failed, use default task3 status")
        traceback.print_exc()
        run_errors.append("task2: {}".format(exc))
        return None


def close_safely(label, close_action):
    try:
        close_action()
    except Exception:
        print("[Main] failed to close {}".format(label))
        traceback.print_exc()


def close_dog(dog):
    dog.stop()
    dog.close()


def main():
    manual_run_time = input_run_time()
    total_started_at = time.perf_counter()
    task_seconds = {}
    run_errors = []
    run_status = "failed"
    dog = None
    task3_runner = None
    vision = VisionManager(show_dashboard_stream=SHOW_TASK2_STREAM)

    try:
        dog = DogControl()
        dog.stand_up()
        dog.close_continue()
        dog.stop()

        print("[Main] start task1")
        with measure_task(task_seconds, "task1"):
            start_yaw_deg = task1.run(
                dog,
                on_navigation_ready=vision.start_dashboard,
            )
            print("[Main] start_yaw_deg={:.3f}".format(start_yaw_deg))
            print("[Main] task1 finished, check yaw once")
            correct_yaw(
                dog,
                start_yaw_deg - 90.0,
                tolerance_deg = 2,
                max_adjust_steps=3,
                stable_need_frames=3,
                sample_count=3,
                settle_s=0.5,
            )
            # time.sleep(2)

        print("[Main] start task2")
        with measure_task(task_seconds, "task2"):
            try:
                dashboard_detector = vision.get_dashboard()
            except Exception as exc:
                run_errors.append("task2 vision: {}".format(exc))
                dashboard_detector = None
            records = run_task2(dog, dashboard_detector, run_errors) if dashboard_detector else None
        status_by_letter = get_task3_status(records)
        print("[Main] task3 status={}".format(status_by_letter))

        print("[Main] start task2_3 bridge")
        with measure_task(task_seconds, "task2_3"):
            task2_3.run(dog, start_yaw_deg=start_yaw_deg, detector=vision.get_dashboard(retry=True))
        print("[Main] keep task2/task2_3 detector for task3")
        dog.revolve_180()

        print("[Main] reset arm before task3")
        with measure_task(task_seconds, "task3"):
            reset_arm()
            # 稍微往右一点
            dog.move(last_time=0.3, vy=25000)
            task3_runner = Task3(
                status_dict=status_by_letter,
                dog=dog,
                vision=vision.take_task3(),
            )
            print("[Main] start task3")
            task3_runner.start()
            task3_runner.run()

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
        close_safely("vision manager", vision.close)
        close_safely("ROS pose reader", close_pose_reader)
        if task3_runner is not None:
            close_safely("task3 runner", task3_runner.close)
        if dog is not None:
            close_safely("dog control", lambda: close_dog(dog))
        close_safely(
            "run logger",
            lambda: append_run_log(
                task_seconds=task_seconds,
                total_seconds=time.perf_counter() - total_started_at,
                status=run_status,
                errors=run_errors,
                run_time=manual_run_time,
            ),
        )


if __name__ == "__main__":
    main()
