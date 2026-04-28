from dog_control_sxh_test import DogControl
import time

if __name__ == "__main__":
    robot = DogControl()
    time.sleep(1)
    robot.start_continue()
    time.sleep(1)
    robot.close_continue()
    print("已经停止")
    #robot.stop()
    time.sleep(0.5)
