import time
from dog_control_sxh_test import DogControl

def walk_task(robot: DogControl):

    # print("站起来")
    # robot.stand_up()
    # time.sleep(2)

    # 中步态向前走
    print("低速步态向前走 2 秒")
    robot.move(last_time=2, vx=20000)
    time.sleep(2)

    # 高步态向前走
    # print("高步态向前走 2 秒")
    # robot.EXmove(last_time=2, vx=2500, case=2)
    # time.sleep(0.5)

    # 转头
    # print("原地右转 90 度")
    # robot.revolve_90()
    # time.sleep(0.5)

    # 再向前走
    # print("中步态向前走 1.5 秒")
    # robot.EXmove(last_time=1.5, vx=2000, case=1)
    # time.sleep(0.5)


    # print("停止")
    # robot.stop()