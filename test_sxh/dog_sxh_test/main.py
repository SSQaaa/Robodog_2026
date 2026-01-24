from dog_control_sxh_test import DogControl
from tasks.walk import walk_task
from tasks.process_three import Process_Three
import time

if __name__ == "__main__":
    robot = DogControl()
    time.sleep(1)
    robot.close_continue()
    robot.stop()
    time.sleep(0.5)
    robot.stand_up()
    time.sleep(0.5)

    #walk_task(robot)
    Process_Three(robot)

    robot.close_continue()
    robot.stop()
    time.sleep(2)