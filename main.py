# -*- coding: utf-8 -*-
import traceback

import task1
import task2
from arm_control import reset_arm
from project_config import DEFAULT_DASHBOARD_STATUS
from task3 import Task3
from tools.motion import DogControl
from tools.vision import resolve_dashboard_status


def main():
    dog = None
    task3_runner = None
    try:

        dog = DogControl()
        dog.stand_up()
        dog.close_continue()
        dog.stop()

        print("[Main] start task1")
        task1.run(dog)

        print("[Main] start task2")
        records = task2.run(dog)
        status_by_letter = resolve_dashboard_status(records, default_status=DEFAULT_DASHBOARD_STATUS)
        print(f"[Main] task3 status={status_by_letter}")

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
        if dog is not None:
            dog.stop()
            dog.close()


if __name__ == "__main__":
    main()
