from dog_control_sxh_test import DogControl
from tasks.walk import walk_task
from tasks.process_three import Process_Three
from tasks.play_football import play_football
import time
from tasks.task2 import task2

if __name__ == "__main__":
    robot = DogControl()
    time.sleep(1)
    robot.close_continue()
    robot.stop()
    time.sleep(0.5)
    #起立之后是原地模式，同时自带了duration为3，也就是说这里的起立之后会停顿3秒左右
    robot.stand_up()
    time.sleep(0.5)

    # 踢球任务
    # play_football(robot, color='orange', show_video=True)
    # 任务二
    task2(robot)

    robot.close_continue()
    robot.stop()
    print("stop")
    time.sleep(2)