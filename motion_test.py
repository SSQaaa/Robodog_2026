# -*- coding: utf-8 -*-
import time

from tools.motion import DogControl


# 自己在这里改速度和时间。
FORWARD_VX = 10000
FORWARD_TIME_S = 0.3

# vy 正数向右，负数向左。
LEFT_VY = 20000
LEFT_TIME_S = 0.3

MOVE_SETTLE_S = 0.3


def main():
    dog = DogControl()
    try:
        # print("[MotionTest] stand up")
        # dog.stand_up()
        # dog.close_continue()
        # dog.stop()
        # time.sleep(0.5)
        dog.move(vx=0, vy=25000, last_time=2.5, duration=MOVE_SETTLE_S)
        dog.stop()
        time.sleep(2)

    #     print("[MotionTest] move forward vx={} time={:.2f}s".format(FORWARD_VX, FORWARD_TIME_S))
    #     dog.move(vx=FORWARD_VX, last_time=FORWARD_TIME_S, duration=MOVE_SETTLE_S)
    #    # dog.stop()
    #     time.sleep(2)
    #     print("[MotionTest] move forward vx={} time={:.2f}s".format(FORWARD_VX, FORWARD_TIME_S))
    #     dog.move(vx=FORWARD_VX, last_time=FORWARD_TIME_S, duration=MOVE_SETTLE_S)
    #     #dog.stop()
    #     time.sleep(2)
    #     print("[MotionTest] move forward vx={} time={:.2f}s".format(FORWARD_VX, FORWARD_TIME_S))
    #     dog.move(vx=FORWARD_VX, last_time=FORWARD_TIME_S, duration=MOVE_SETTLE_S)
    #     #dog.stop()
    #     time.sleep(2)
    #     print("[MotionTest] move forward vx={} time={:.2f}s".format(FORWARD_VX, FORWARD_TIME_S))
    #     dog.move(vx=FORWARD_VX, last_time=FORWARD_TIME_S, duration=MOVE_SETTLE_S)
    #     #dog.stop()
    #     time.sleep(2)
    #     print("[MotionTest] move forward vx={} time={:.2f}s".format(FORWARD_VX, FORWARD_TIME_S))
    #     dog.move(vx=FORWARD_VX, last_time=FORWARD_TIME_S, duration=MOVE_SETTLE_S)
    #     #dog.stop()
    #     time.sleep(2)

        # print("[MotionTest] move left vy={} time={:.2f}s".format(LEFT_VY, LEFT_TIME_S))
        # dog.move(vy=LEFT_VY, last_time=LEFT_TIME_S, duration=MOVE_SETTLE_S)
        # dog.stop()
        # time.sleep(0.5)
        # print("[MotionTest] move left vy={} time={:.2f}s".format(LEFT_VY, LEFT_TIME_S))
        # dog.move(vy=LEFT_VY, last_time=LEFT_TIME_S, duration=MOVE_SETTLE_S)
        # dog.stop()
        # time.sleep(0.5)
        # print("[MotionTest] move left vy={} time={:.2f}s".format(LEFT_VY, LEFT_TIME_S))
        # dog.move(vy=LEFT_VY, last_time=LEFT_TIME_S, duration=MOVE_SETTLE_S)
        # dog.stop()
        # time.sleep(0.5)
        # print("[MotionTest] move left vy={} time={:.2f}s".format(LEFT_VY, LEFT_TIME_S))
        # dog.move(vy=LEFT_VY, last_time=LEFT_TIME_S, duration=MOVE_SETTLE_S)
        # dog.stop()
        # time.sleep(0.5)

        print("[MotionTest] done")
    finally:
        dog.stop()
        dog.close()


if __name__ == "__main__":
    main()
