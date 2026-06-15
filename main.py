# -*- coding: utf-8 -*-
import traceback

import task1
import task2
import task2_3
from arm_control import reset_arm
from project_config import DEFAULT_DASHBOARD_STATUS
from task3 import Task3
from tools.motion import DogControl
from tools.vision import resolve_dashboard_status


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
    dog = None
    task2_detector = None
    task2_detector_preload = None
    task3_runner = None
    try:

        dog = DogControl()
        dog.stand_up()
        dog.close_continue()
        dog.stop()

        # print("[Main] start task1")
        # task1.run(dog)

        print("[Main] start task2")
        task2_detector = task2_detector_preload.result()
        task2_detector_preload = None
        try:
            records = task2.run(dog, detector=task2_detector, close_detector=False)
        except Exception:
            print("[Main] task2 failed, use default task3 status")
            traceback.print_exc()
            records = None
        status_by_letter = get_task3_status(records)
        print(f"[Main] task3 status={status_by_letter}")

        print("[Main] start task2_3 bridge")
        task2_3.task2_3(dog, task2_detector)
        task2_detector.close()
        task2_detector = None

        print("[Main] reset arm before task3")
        reset_arm()
        print("[Main] start task3")
        task3_runner = Task3(status_dict=status_by_letter, dog=dog)
        task3_runner.start()
        task3_runner.run()

        print("[Main] all tasks finished")
    except Exception:
        print("[Main] failed, stopping robot")
        traceback.print_exc()
        if dog is not None:
            dog.stop()
        raise
    finally:
        if task3_runner is not None:
            task3_runner.close()
        if task2_detector is not None:
            task2_detector.close()
        if task2_detector_preload is not None:
            task2_detector_preload.close_if_ready()
        if dog is not None:
            dog.stop()
            dog.close()


if __name__ == "__main__":
    main()
