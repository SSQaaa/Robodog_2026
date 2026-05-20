from dog_control_sxh_test import DogControl
import argparse
import time
from tasks.task2_new import task2_new   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream", action="store_true", help="显示任务二检测画面")
    args = parser.parse_args()

    robot = DogControl()
    time.sleep(1)
    robot.close_continue()
    robot.stop()
    time.sleep(0.5)
    #起立之后是原地模式，同时自带了duration为3，也就是说这里的起立之后会停顿3秒左右
    robot.stand_up()
    time.sleep(0.5)

    # 任务二
    # task2(robot)
    # 新的任务二
    task2_new(robot, show_stream=args.stream)

    robot.close_continue()
    robot.stop()
    print("stop")
    time.sleep(2)
